import sys
import os
import numpy as np
import argparse
from types import MethodType
import pandas as pd

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
parser.add_argument('--alpha', type=float, default=128, help='alpha')
parser.add_argument('--beta', type=float, default=8, help='beta')
parser.add_argument('--num_subsets', type=int, default=6, help='number of subsets')

parser.add_argument('--iterations', type=int, default=60, help='max iterations')
parser.add_argument('--update_interval', type=int, default=6, help='update interval')
parser.add_argument('--relaxation_eta', type=float, default=0.01, help='relaxation eta')

parser.add_argument('--data_path', type=str, default="/home/sam/data/phantom_data/for_cluster/", help='data path')
parser.add_argument('--output_path', type=str, default="/home/sam/working/BSREM_PSMR2024/results/test", help='output path')
parser.add_argument('--source_path', type=str, default='/home/sam/working/BSREM_PSMR2024/src', help='source path')
parser.add_argument('--working_path', type=str, default='/home/sam/working/BSREM_PSMR2024/tmp', help='working path')

parser.add_argument('--gpu', type=bool, default=False, help='use gpu')

args = parser.parse_args()

# Imports from my stuff and SIRF contribs
sys.path.insert(0, args.source_path)
import partitioner.partitioner as partitioner
from BSREM.BSREM import BSREMmm, BSREM2, BSREMmm_of
from structural_priors.tmp_classes import (OperatorCompositionFunction,
                                                   ZoomOperator, CompositionOperator,
                                                    NiftyResampleOperator,
                                                    FairL21Norm,
                                                )

from structural_priors.Operator import Operator
from structural_priors.Function import Function
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

# Class definitions to put DataContainer into Numpy array and back
class NumpyBlockDataContainer(Operator):

    def __init__(self, domain_geometry, operator):

        self.domain_geometry = domain_geometry
        self.array = np.stack([np.zeros((d.shape)) for d in domain_geometry.containers], axis=-1)
        self.operator = operator

    def direct(self, x, out=None):
        x_arr = np.stack([d.as_array() for d in x.containers], axis=-1)
        return self.operator.direct(x_arr)

    def adjoint(self, x, out=None):
        x_arr = self.operator.adjoint(x)
        res = self.domain_geometry.clone()
        for i, r in enumerate(res.containers):
            r.fill(x_arr[...,i])
        return res  

class NumpyDataContainer(Operator):

    def __init__(self, domain_geometry, operator):

        self.domain_geometry = domain_geometry
        self.array = domain_geometry.as_array()
        self.operator = operator

    def direct(self, x, out=None):
        x_arr = x.as_array()
        return self.operator.direct(x_arr)

    def adjoint(self, x, out=None):
        res = self.domain_geometry.clone()
        res.fill(self.operator.adjoint(x))
        return res
    
class SIRFBlockFunction(Function):
    
    def __init__(self, functions):
        self.functions = functions
        
    def __call__(self, x):
        return sum(f(el) for f, el in zip(self.functions, x.containers))
    
    def get_subset_gradient(self, x, subset=0):
        return BlockDataContainer(*[f.get_subset_gradient(el, subset) for f, el in zip(self.functions, x.containers)])   
    
    def get_subset_sensitivity(self, subset_num):
        return BlockDataContainer(*[f.get_subset_sensitivity(subset_num) for f in self.functions])

def get_filters():
    cyl, gauss = TruncateToCylinderProcessor(), SeparableGaussianImageFilter()
    cyl.set_strictly_less_than_radius(True)
    gauss.set_fwhms((5,5,5))
    return cyl, gauss

def get_pet_data(path):

    pet_data = {}
    pet_data["acquisition_data"] = AcquisitionData(path + "PET/projdata_bed0.hs")
    pet_data["additive"] = AcquisitionData(path + "PET/additive3d_bed0_nonan.hs")
    pet_data["normalisation"] = AcquisitionData(path + "PET/inv_normacfprojdata_bed0.hs")
    pet_data["initial_image"] = ImageData(path + "PET/pet_osem_20.hv").maximum(0)

    return pet_data

def get_spect_data(path):

    spect_data = {}
    spect_data["acquisition_data"] = AcquisitionData(path + "SPECT/peak_1_projdata__f1g1d0b0.hs")
    spect_data["additive"] = AcquisitionData(path + "SPECT/simind_scatter_osem_555_full.hs")
    spect_data["attenuation"] = ImageData(path + "SPECT/umap_zoomed.hv")
    spect_data["initial_image"] = ImageData(path + "SPECT/spect_osem_20.hv").maximum(0)

    return spect_data

def get_zoom_transform(data_path, filename, zoom_operator, template_image):
    transform =  AffineTransformation(os.path.join(data_path, "Registration", filename))
    resampler = NiftyResampleOperator(template_image, template_image, transform)
    return CompositionOperator([zoom_operator, resampler])

def get_pet_am(pet_data, gpu=False):
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

def get_spect_am(spect_data):
    spect_am_mat = SPECTUBMatrix()
    spect_am_mat.set_attenuation_image(spect_data["attenuation"])
    spect_am_mat.set_keep_all_views_in_cache(True)
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
    spect_am.set_additive_term(spect_data["additive"])
    #spect_am.set_up(spect_data["acquisition_data"], spect_data["initial_image"])
    return spect_am

def get_objective_function(data, acq_model, initial_image, num_subsets):
    
    obj_fun = make_Poisson_loglikelihood(data)
    obj_fun.set_acquisition_model(acq_model)
    obj_fun.set_num_subsets(num_subsets)
    obj_fun.set_up(initial_image)
    
    return obj_fun

def get_vectorial_tv(bo, ct, alpha, beta, initial_estimates):
    vtv = create_vectorial_total_variation(smoothing_function='fair', gpu=False)
    jac = NumpyBlockDataContainer(bo.direct(initial_estimates),Jacobian(anatomical=ct.as_array(), voxel_sizes=ct.voxel_sizes(), weights=[alpha, beta]))
    jac_co = CompositionOperator([bo, jac])
    jac_co.range_geometry = lambda: initial_estimates
    return OperatorCompositionFunction(vtv, jac_co)

def get_tv(operator, ct, regularisation_parameter, initial_estimate):
    dtv = regularisation_parameter * FairL21Norm(delta=1e-6)
    grad = NumpyDataContainer(operator.direct(initial_estimate), DirectionalGradient(anatomical=ct.as_array(), voxel_sizes=ct.voxel_sizes()))
    grad_co = CompositionOperator([operator, grad])
    grad_co.range_geometry = lambda: initial_estimate
    return OperatorCompositionFunction(dtv, grad_co)    

# Change to working directory - this is where the tmp_ files will be saved
os.chdir(args.working_path)

def main(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cyl, gauss = get_filters()
    ct = ImageData(args.data_path + "CT/ct_zoomed_pet.hv")
    # normalise the CT image
    ct+=(-ct).max()
    ct/=ct.max()    

    if args.modality.lower() == "pet" or args.modality.lower() == "both":

        pet_data  = get_pet_data(args.data_path)
        cyl.apply(pet_data["initial_image"])

        pet2ct_zoom = ZoomOperator(ct, pet_data["initial_image"])
        pet2ct = get_zoom_transform(args.data_path, "pet2spect_zoomed_pet.txt", pet2ct_zoom, ct)
        pet_am = get_pet_am(pet_data,  args.gpu)
        pet_am.direct = lambda x: pet_am.forward(x)
        pet_am.adjoint = lambda x: pet_am.backward(x)
        
        pet_obj_fun = get_objective_function(pet_data["acquisition_data"], pet_am, pet_data["initial_image"], args.num_subsets)

    if args.modality.lower() == "spect" or args.modality.lower() == "both":

        spect_data  = get_spect_data(args.data_path)
        cyl.apply(spect_data["initial_image"])

        spect2ct_zoom = ZoomOperator(ct, spect_data["initial_image"])
        spect2ct = get_zoom_transform(args.data_path, "spect2ct_zoomed_pet.txt", spect2ct_zoom, ct)
        spect_am = get_spect_am(spect_data)
        spect_am.direct = lambda x: spect_am.forward(x)
        spect_am.adjoint = lambda x: spect_am.backward(x)
        
        spect_obj_fun = get_objective_function(spect_data["acquisition_data"], spect_am, spect_data["initial_image"], args.num_subsets)
    
    if args.modality.lower() == "both":
        zero_pet2ct = ZeroOperator(pet_data["initial_image"], ct)
        zero_spect2ct = ZeroOperator(spect_data["initial_image"], ct)

        bo = BlockOperator(pet2ct, zero_spect2ct,
                            zero_pet2ct, spect2ct, 
                            shape = (2,2))

        initial_estimates = BlockDataContainer(pet_data["initial_image"], spect_data["initial_image"])

        prior = get_vectorial_tv(bo, ct, args.alpha, args.beta, initial_estimates)


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
        prior = get_tv(pet2ct, ct, args.alpha, pet_data["initial_image"])
        obj_fun = pet_obj_fun

        initial = pet_data["initial_image"]

    elif args.modality.lower() == "spect":
        acquisition_model = spect_am
        data = spect_data["acquisition_data"]
        prior = get_tv(spect2ct, ct, args.beta, spect_data["initial_image"])
        obj_fun = spect_obj_fun

        initial = spect_data["initial_image"]
        
    acquisition_model.is_linear = MethodType(lambda self: True, acquisition_model)

    if args.modality.lower() == "both":
        bsrem=BSREMmm_of(data, SIRFBlockFunction([pet_obj_fun, spect_obj_fun]), prior, 
                         initial=initial, initial_step_size=1, relaxation_eta=args.relaxation_eta, 
                         update_objective_interval=args.update_interval, save_path=args.output_path)
    else:
        bsrem = BSREMmm_of(data, obj_fun, prior, initial=initial, initial_step_size=1, 
                           relaxation_eta=args.relaxation_eta, 
                           update_objective_interval=args.update_interval, save_path=args.output_path)

    bsrem.max_iteration=args.iterations
    bsrem.run(args.iterations, verbose=2)

    return bsrem

if __name__ == "__main__":

    _ = MessageRedirector()

    bsrem = main(args)

    if isinstance(bsrem.x, DataContainer):
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

    for file in os.listdir(os.getcwd()):
        if file.startswith("tmp_") and (file.endswith(".s") or file.endswith(".hs")):
            os.remove(file)

    print("Done")

