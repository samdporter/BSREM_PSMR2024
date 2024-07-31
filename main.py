import sys
import os
import numpy as np
import argparse
from types import MethodType
import pandas as pd
from numbers import Number
import shutil

# import gaussian filter
from scipy.ndimage import gaussian_filter

# SIRF imports
from sirf.STIR import (ImageData, AcquisitionData, 
                       AcquisitionModelUsingMatrix, 
                       AcquisitionModelUsingParallelproj, 
                       AcquisitionSensitivityModel, 
                       SPECTUBMatrix,MessageRedirector,
                       TruncateToCylinderProcessor, SeparableGaussianImageFilter,
                       AcquisitionModelUsingRayTracingMatrix,
                       make_Poisson_loglikelihood,
                       )
from sirf.Reg import AffineTransformation

# CIL imports
from cil.framework import BlockDataContainer, DataContainer
from cil.optimisation.operators import BlockOperator, ZeroOperator

parser = argparse.ArgumentParser(description='BSREM')

parser.add_argument('--modality', type=str, default='both', help='modality - can be pet, spect or both')
parser.add_argument('--alpha', type=float, default=256, help='alpha')
parser.add_argument('--beta', type=float, default=1, help='beta')
parser.add_argument('--delta', type=float, default=1e-6, help='delta')
# num_subsets can be an integer or a string of two integers separated by a comma
parser.add_argument('--num_subsets', type=str, default="12", help='number of subsets')


parser.add_argument('--iterations', type=int, default=240, help='max iterations')
parser.add_argument('--update_interval', type=int, default=12, help='update interval')
parser.add_argument('--relaxation_eta', type=float, default=0.1, help='relaxation eta')

parser.add_argument('--data_path', type=str, default="/home/sam/data/phantom_data/for_cluster/", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR_MIC_2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR_MIC_2024/tmp', help='working path')
parser.add_argument('--save_images', type=bool, default=True, help='save images')

# set numpy seed - None if not set
parser.add_argument('--seed', type=int, default=None, help='numpy seed')

parser.add_argument('--stochastic', action='store_true', help='Enables stochastic processing')
parser.add_argument('--svrg', action='store_true', help='Enables SVRG')
parser.add_argument('--saga', action='store_true', help='Enables SAGA')
parser.add_argument('--with_replacement', action='store_true', help='Enables replacement')
parser.add_argument('--single_modality_update', action='store_true', help='Enables single modality update')
parser.add_argument('--prior_is_subset', action='store_true', help='Sets prior as subset')
parser.add_argument('--gpu', action='store_true', help='Enables GPU')
parser.add_argument('--keep_all_views_in_cache', action='store_true', help='Keep all views in cache')

args = parser.parse_args()

# Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
from BSREM.BSREM import BSREMmm_of
from structural_priors.tmp_classes import (OperatorCompositionFunction,
                                                   ZoomOperator, CompositionOperator,
                                                    NiftyResampleOperator,
                                                    FairL21Norm,
                                                )

from structural_priors.Operator import Operator, NumpyDataContainer, NumpyBlockDataContainer
from structural_priors.Function import Function, SIRFBlockFunction
Operator.is_linear = lambda self: True
from structural_priors.Gradients import DirectionalGradient
from structural_priors.VTV import create_vectorial_total_variation
from structural_priors.Gradients import Jacobian

# Monkey patching
BlockOperator.forward = lambda self, x: self.direct(x)
BlockOperator.backward = lambda self, x: self.adjoint(x)

BlockDataContainer.get_uniform_copy = lambda self, n: BlockDataContainer(*[x.clone().fill(n) for x in self.containers])
BlockDataContainer.max = lambda self: max(d.max() for d in self.containers)

ZeroOperator.backward = lambda self, x: self.adjoint(x)
ZeroOperator.forward = lambda self, x: self.direct(x)

def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((7,7,7))
    return cyl, gauss

def get_pet_data(path):

    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(os.path.join(path,  "PET/projdata_bed0.hs"))
    pet_data["additive"] = AcquisitionData(os.path.join(path,  "PET/additive3d_bed0_nonan.hs"))
    pet_data["normalisation"] = AcquisitionData(os.path.join(path,  "PET/inv_normacfprojdata_bed0.hs"))
    pet_data["initial_image"] = ImageData(os.path.join(path,  "PET/pet_osem_20.hv")).maximum(0)

    return pet_data

def get_spect_data(path):

    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(os.path.join(path,  "SPECT/peak_1_projdata__f1g1d0b0.hs"))
    spect_data["additive"] = AcquisitionData(os.path.join(path,  "SPECT/simind_scatter_osem_555_full_smoothed.hs"))
    spect_data["attenuation"] = ImageData(os.path.join(path,  "SPECT/umap_zoomed.hv"))
    spect_data["initial_image"] = ImageData(os.path.join(path,  "SPECT/spect_osem_20.hv")).maximum(0)

    return spect_data

def get_zoom_transform(data_path, filename, zoom_operator, template_image):
    transform =  AffineTransformation(os.path.join(data_path, "Registration", filename))
    resampler = NiftyResampleOperator(template_image, template_image, transform)
    return CompositionOperator([zoom_operator, resampler])

def get_pet_am(pet_data, gpu):
    if gpu:
        pet_am = AcquisitionModelUsingParallelproj()
    else:
        pet_am = AcquisitionModelUsingRayTracingMatrix()
        pet_am.set_num_tangential_LORs(10)
    asm = AcquisitionSensitivityModel(pet_data["normalisation"])
    pet_am.set_acquisition_sensitivity(asm)
    pet_am.set_additive_term(pet_data["additive"])
    # using adjoint(forard(image)) & STIR find_fwhm_in_image
    # 1cm FWHM from NEMA 2001 (Mediso AnyScan specificaitons)
    # operations applied one after the other to find total FWHM
    pet_psf = SeparableGaussianImageFilter()
    pet_psf.set_fwhms((4.2,4.1,4.1)) 
    pet_am.set_image_data_processor(pet_psf)
    #pet_am.set_up(pet_data["acquisition_data"], pet_data["initial_image"])
    return pet_am

def get_spect_am(spect_data, keep_all_views_in_cache=False):
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_attenuation_image(spect_data["attenuation"])
    spect_am_mat.set_keep_all_views_in_cache(keep_all_views_in_cache)
    # using Tc99m (140 keV) AnyScan measured resolution modelling
    # close enough to 150 keV Y90
    #spect_mat.set_resolution_model(1.81534, 0.02148, False) 
    # spect_mat.set_resolution_model(0.9323, 0.03, False) 
    #spect_am = AcquisitionModelUsingMatrix(spect_mat)
    #spect_psf = SeparableGaussianImageFilter()
    # using gaussians estimated from Y90 data (see notebook)
    # /home/sam/working/simulated_data/data/data.ipynb
    #spect_psf.set_fwhms((6.61, 6.61, 6.61))
    #spect_am.set_image_data_processor(spect_psf)
    spect_am_mat.set_resolution_model(0.93,0.03, False) 
    spect_am = AcquisitionModelUsingMatrix(spect_am_mat)
    spect_am.set_additive_term(spect_data["additive"]) #TODO: change back
    #spect_am.set_up(spect_data["acquisition_data"], spect_data["initial_image"])
    return spect_am

def get_objective_function(data, acq_model, initial_image, num_subsets):
    
    obj_fun = make_Poisson_loglikelihood(data)
    obj_fun.set_acquisition_model(acq_model)
    obj_fun.set_num_subsets(num_subsets)
    obj_fun.set_up(initial_image)
    
    return obj_fun

def get_vectorial_tv(bo, ct, alpha, beta, initial_estimates, delta, gpu=False):
    vtv = create_vectorial_total_variation(smoothing_function='fair', eps=delta, gpu=gpu)
    jac = NumpyBlockDataContainer(bo.direct(initial_estimates),Jacobian(anatomical=ct.as_array(), voxel_sizes=ct.voxel_sizes(), weights=[alpha, beta], gpu=gpu))
    jac_co = CompositionOperator([bo, jac])
    jac_co.range_geometry = lambda: initial_estimates
    return OperatorCompositionFunction(vtv, jac_co)

def get_tv(operator, ct, regularisation_parameter, initial_estimate, delta):
    dtv = regularisation_parameter * FairL21Norm(delta=delta)
    grad = NumpyDataContainer(operator.direct(initial_estimate), DirectionalGradient(anatomical=ct.as_array(), voxel_sizes=ct.voxel_sizes()))
    grad_co = CompositionOperator([operator, grad])
    grad_co.range_geometry = lambda: initial_estimate
    return OperatorCompositionFunction(dtv, grad_co)    

# Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

def main(args):

    # if single_modality_update is False, num_subsets must be integer
    if not args.single_modality_update:
        try:
            args.num_subsets = int(args.num_subsets)
        except:
            raise ValueError("num_subsets must be an integer if single_modality_update is False")
    if isinstance(args.num_subsets, Number):
        pet_num_subsets = int(args.num_subsets)
        spect_num_subsets = int(args.num_subsets)
    elif isinstance(args.num_subsets, str):
        num_subsets = args.num_subsets
        subset_list = num_subsets.split(",")
        # if list is lenght one, set both to the same value
        if len(subset_list) == 1:
            pet_num_subsets = int(subset_list[0])
            spect_num_subsets = int(subset_list[0])
        else:
            pet_num_subsets = int(subset_list[0])
            spect_num_subsets = int(subset_list[1])

    cyl, gauss, = get_filters()
    ct = ImageData(os.path.join(args.data_path, "CT/ct_zoomed_pet.hv"))
    # normalise the CT image
    ct+=(-ct).max()
    ct/=ct.max()    

    if args.modality.lower() == "pet" or args.modality.lower() == "both":

        pet_data  = get_pet_data(args.data_path)
        cyl.apply(pet_data["initial_image"])
        #gauss.apply(pet_data["initial_image"])
        pet_data["initial_image"].write("initial_image_0.hv")

        pet2ct_zoom = ZoomOperator(ct, pet_data["initial_image"])
        pet2ct = get_zoom_transform(args.data_path, "pet2spect_zoomed_pet.txt", pet2ct_zoom, ct)
        pet_am = get_pet_am(pet_data,  args.gpu)
        pet_am.direct = lambda x: pet_am.forward(x)
        pet_am.adjoint = lambda x: pet_am.backward(x)
        
        pet_obj_fun = get_objective_function(pet_data["acquisition_data"], pet_am, pet_data["initial_image"], pet_num_subsets)

    if args.modality.lower() == "spect" or args.modality.lower() == "both":

        spect_data  = get_spect_data(args.data_path)
        cyl.apply(spect_data["initial_image"])
        #gauss.apply(spect_data["initial_image"])
        spect_data["initial_image"].write("initial_image_1.hv")

        spect2ct_zoom = ZoomOperator(ct, spect_data["initial_image"])
        spect2ct = get_zoom_transform(args.data_path, "spect2ct_zoomed_pet.txt", spect2ct_zoom, ct)
        spect_am = get_spect_am(spect_data, args.keep_all_views_in_cache)
        spect_am.direct = lambda x: spect_am.forward(x)
        spect_am.adjoint = lambda x: spect_am.backward(x)
        
        spect_obj_fun = get_objective_function(spect_data["acquisition_data"], spect_am, spect_data["initial_image"], spect_num_subsets)
    
    if args.modality.lower() == "both":
        zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
        zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

        bo = BlockOperator(pet2ct, zero_spect2ct,
                            zero_pet2ct, spect2ct, 
                            shape = (2,2))

        initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])

        prior = get_vectorial_tv(bo, ct, args.alpha, args.beta, initial_estimates, delta=args.delta, gpu=False)


        pet2spect_zero = ZeroOperator(pet_data["initial_image"], spect_data["acquisition_data"])
        spect2pet_zero = ZeroOperator(spect_data["initial_image"], pet_data["acquisition_data"])

        acquisition_model = BlockOperator(pet_am, spect2pet_zero,
                                            pet2spect_zero, spect_am,
                                        shape=(2,2)) 
        data = BlockDataContainer(pet_data["acquisition_data"], spect_data["acquisition_data"])

        initial = initial_estimates
        
        
    elif args.modality.lower() == "pet":
        acquisition_model = pet_am
        data = pet_data["acquisition_data"]
        prior = get_tv(pet2ct, ct, args.alpha, pet_data["initial_image"], delta=args.delta)
        obj_fun = pet_obj_fun

        initial = pet_data["initial_image"]

    elif args.modality.lower() == "spect":
        acquisition_model = spect_am
        data = spect_data["acquisition_data"]
        prior = get_tv(spect2ct, ct, args.beta, spect_data["initial_image"], delta=args.delta)
        obj_fun = spect_obj_fun

        initial = spect_data["initial_image"]
        
    acquisition_model.is_linear = MethodType(lambda self: True, acquisition_model)

    if args.modality.lower() == "both":
        bsrem=BSREMmm_of(SIRFBlockFunction([pet_obj_fun, spect_obj_fun]), prior, 
                         initial=initial, initial_step_size=1, relaxation_eta=args.relaxation_eta, 
                         update_objective_interval=args.update_interval, save_path=args.working_path,
                         stochastic=args.stochastic, svrg=args.svrg, saga=args.saga, with_replacement=args.with_replacement,
                         single_modality_update=args.single_modality_update, save_images = args.save_images,
                         prior_is_subset=args.prior_is_subset, update_max=100*initial.max())
    else:
        bsrem = BSREMmm_of(obj_fun, prior, initial=initial, initial_step_size=1, 
                           relaxation_eta=args.relaxation_eta, 
                           update_objective_interval=args.update_interval, save_path=args.working_path,
                           stochastic=args.stochastic, svrg=args.svrg, saga=args.saga, with_replacement=args.with_replacement,
                           single_modality_update=False, save_images=args.save_images,
                           prior_is_subset=args.prior_is_subset, update_max=100*initial.max())

    bsrem.max_iteration=args.iterations
    bsrem.run(args.iterations, verbose=2)

    return bsrem

if __name__ == "__main__":

    _ = MessageRedirector()
    
    # create dataframe of all args
    df_args = pd.DataFrame([vars(args)])
    df_args.to_csv(os.path.join(args.output_path, "args.csv"))

    # print all args
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    bsrem = main(args)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if isinstance(bsrem.x, ImageData):
        bsrem.x.write(os.path.join(args.output_path, f"bsrem_a_{args.alpha}_b_{args.beta}.hv"))
    elif isinstance(bsrem.x, BlockDataContainer):
        for i, el in enumerate(bsrem.x.containers):
            el.write(os.path.join(args.output_path, f"bsrem_modality_{i}_a_{args.alpha}_b_{args.beta}.hv"))
    df = pd.DataFrame(l[0] for l in bsrem.loss) 
    df.to_csv(os.path.join(args.output_path, f"bsrem_objective_a_{args.alpha}_b_{args.beta}.csv"))

    df_data = pd.DataFrame([l[1] for l in bsrem.loss])
    df_prior = pd.DataFrame([l[2] for l in bsrem.loss])

    df_data.to_csv(os.path.join(args.output_path, f"bsrem_data_a_{args.alpha}_b_{args.beta}.csv"))
    df_prior.to_csv(os.path.join(args.output_path, f"bsrem_prior_a_{args.alpha}_b_{args.beta}.csv"))

    # stick in one file
    df_full = pd.concat([df, df_data, df_prior], axis=1)
    # add column names
    df_full.columns = ["Objective", "Data", "Prior"]
    df_full.to_csv(os.path.join(args.output_path, f"bsrem_full_a_{args.alpha}_b_{args.beta}.csv"))

    for file in os.listdir(args.working_path):
        if file.startswith("tmp_") and (file.endswith(".s") or file.endswith(".hs")):
            os.remove(file)

    # move any leftover files to output path
    for file in os.listdir(args.working_path):
        if (file.endswith(".hv") or file.endswith(".v") or file.endswith(".ahv")):
            print(f"Moving {file}")
            shutil.move(os.path.join(args.working_path, file), os.path.join(args.output_path, file))

    print("Done")

