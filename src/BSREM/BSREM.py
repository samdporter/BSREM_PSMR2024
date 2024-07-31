#
# SPDX-License-Identifier: Apache-2.0
#
# Classes implementing the BSREM algorithm in sirf.STIR
#
# Authors:  Kris Thielemans, Sam Porter
#
# Copyright 2024 University College London

import numpy
import os
import sirf.STIR as STIR
from sirf.Utilities import examples_data_path

from cil.optimisation.algorithms import Algorithm 
from cil.framework import DataContainer, BlockDataContainer

import time
from contextlib import contextmanager

# Define the context manager
@contextmanager
def timing(label: str):
    t0 = time.time()
    yield lambda: None  # This yields a do-nothing function to satisfy the 'with' syntax
    t1 = time.time()
    print(f'{label}: {t1 - t0:.6f} seconds')


class BSREMSkeleton(Algorithm):
    def __init__(self, initial, initial_step_size, num_subsets,
                 relaxation_eta, stochastic=False, with_replacement=False, 
                 save_images=True, prior_is_subset=False, update_max = 100, probabilities = None,
                  **kwargs):

        super(BSREMSkeleton, self).__init__(**kwargs)

        # FOV filter to avoid edge artifacts
        self.FOV_filter = DC_Filter()

        # initialise image estimate
        self.x = initial.copy().maximum(0)
        self.FOV_filter(self.x)

        # parameters controlling step size
        self.initial_step_size = initial_step_size
        self.relaxation_eta = relaxation_eta

        # switches for various subset management techniques
        self.stochastic = stochastic
        self.with_replacement = with_replacement
        self.prior_is_subset = prior_is_subset
        
        self.num_subsets = num_subsets
        self.update_max = update_max

        # compute small number to add to image in preconditioner
        # don't make it too small as otherwise the algorithm cannot recover from zeroes.
        self.eps = initial.max()/1e6
        self.limit_size(self.x)
        self.average_sensitivity = initial.get_uniform_copy(0)
        print('Computing average sensitivity')
        for s in range(self.num_subsets):
            self.average_sensitivity += self.subset_sensitivity(s)
        # add a small number to avoid division by zero in the preconditioner
        self.average_sensitivity /= self.num_subsets
        self.average_sensitivity.maximum(self.eps, out=self.average_sensitivity)
        print('Done computing average sensitivity')

        # inititalise subset for ordered so we start from the beginning
        self.subset = -1
        # If we want to only sample from one subset per epoch, we need to keep track
        self.used_subsets = set() 

        # initialise iteration parameter
        self.iteration = 0
        
        # Do we want to save images?
        self.save_images = save_images

        self.probabilities = probabilities

        self.configured = True

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, subset_num):
        raise NotImplementedError

    def epoch(self):
        # self-explanatory
        return self.iteration // self.num_subsets

    def step_size(self):
        # reduce step size once per epoch
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())
        #return self.initial_step_size * (1-self.relaxation_eta)**self.epoch() # I think I like this version more

    def limit_size(self, x):
        """
        This is horrible but necessary because some bugs somehere (possibly STIR)
        They sory themselves out eventually, but this is a quick fix so I can compare objective values
        """
        if isinstance(x, BlockDataContainer):

            for i, el in enumerate(x.containers):
                update_arr = el.as_array()
                if i ==1:
                    update_arr[:11]=0
                    update_arr[116:]=0
                el.fill(update_arr)
        else:
            update_arr = x.as_array()
            update_arr[update_arr > self.update_max] = self.eps
            x.fill(update_arr)
            
    def calculate_preconditioner(self):
        return self.x / self.average_sensitivity
        
    def update(self):

        # if we want to and it's the end of an epoch, lets save our images
        if  self.iteration % self.update_objective_interval == 0 and self.save_images:
            self.add_images()

        # choose our subset base on subset management strategy
        self.subset = self.choose_subset()

        # calculate our subset gradient for th eimage update
        g = self.subset_gradient(self.subset)

        # calculate our image update
        self.x_update = self.step_size() * self.calculate_preconditioner() * g
        self.limit_size(self.x_update) # necessary because of some bugs somewhere

        self.FOV_filter(self.x_update)
        self.x += self.x_update
        self.x.maximum(self.eps, out=self.x)
            
    def choose_subset(self):
        """Selects the next subset based on configuration parameters."""
        if not self.stochastic:
            return (self.subset + 1) % self.num_subsets  # Deterministic selection, cycle through subsets

        if self.with_replacement:
            return self._choose_with_replacement()
        
        return self._choose_without_replacement()

    def _choose_with_replacement(self):
        """Choose a subset with replacement."""
        if self.probabilities is None:
            return numpy.random.randint(self.num_subsets)
        else:
            available_subsets = list(range(self.num_subsets))
            return numpy.random.choices(available_subsets, self.probabilities)

    def _choose_without_replacement(self):
        """Choose a subset without replacement, ensuring no repeat until all subsets are used."""
        if len(self.used_subsets) >= self.num_subsets:
            self.used_subsets.clear()  # Reset when all subsets have been used
        while True:
            subset = numpy.random.randint(self.num_subsets)
            if subset not in self.used_subsets:
                self.used_subsets.add(subset)
                return subset

    def add_images(self):
        # store images here (not yet using callback)
        self.images.append(self.x.copy())

    # next is needed for CIL 21.3, but we don't do any calculation to save time
    def update_objective(self):
        self.loss.append(0)

def apply_cylinder_mask(image, margin_y, margin_x):
    """ Sets elements outside a cylinder along the z-axis to zero. """

    array = image.as_array()

    z_dim, y_dim, x_dim = array.shape

    # Determine the radius and center of the cylinder
    radius_y = (y_dim - 2 * margin_y) / 2
    radius_x = (x_dim - 2 * margin_x) / 2
    radius = min(radius_y, radius_x)
    
    center_y = y_dim // 2
    center_x = x_dim // 2

    # Create a grid of coordinates
    y, x = numpy.ogrid[:y_dim, :x_dim]

    # Compute the distance from the center for each point in the (y, x) plane
    distance_from_center = numpy.sqrt((y - center_y)**2 + (x - center_x)**2)

    # Create a mask for the cylinder
    cylinder_mask = distance_from_center <= radius

    # Expand the mask to the third dimension (z-axis)
    full_mask = numpy.repeat(cylinder_mask[numpy.newaxis, :, :], z_dim, axis=0)

    # Apply the mask to the array
    masked_array = array * full_mask

    image.fill(masked_array)

    
class DC_Filter():

    def __init__(self, margin=10):
        self.FOV_filter = STIR.TruncateToCylinderProcessor()
        self.FOV_filter.set_strictly_less_than_radius(True)

        self.margin = margin

    def __call__(self, x):
        #self.apply_cylinder_mask(x)
        self.apply(x)

    def apply(self, x):
        if isinstance(x, BlockDataContainer):
            for i, el in enumerate(x.containers):
                if i == 0:
                    self.FOV_filter.apply(el)
        else:
            self.FOV_filter.apply(x)

    def apply_cylinder_mask(self, image):
        if isinstance(image, BlockDataContainer):
            for i, el in enumerate(image.containers):
                if i == 0:
                    apply_cylinder_mask(el, margin_y=self.margin, margin_x=self.margin)
        else:
            apply_cylinder_mask(image, margin_y=self.margin, margin_x=self.margin)
        
class BSREMmm_of(BSREMSkeleton):
    """
    BSREM implementation using sirf.STIR objective functions.

    Parameters:
        obj_fun: Objective function used in reconstruction
        prior: Prior knowledge or regularization term
        initial: Initial guess for the reconstruction
        initial_step_size: Initial step size for the optimization
        relaxation_eta: Step size relaxation factor (per epoch)
        save_path: Directory path to save output images
        svrg: Boolean indicating whether to use Stochastic Variance Reduced Gradient (SVRG)
        **kwargs: Additional keyword arguments
    """
    def __init__(self, obj_fun, prior, initial, initial_step_size=1, relaxation_eta=0, save_path='', 
                 svrg=False, saga=False, stochastic=False, with_replacement=False, single_modality_update=False, 
                 save_images =True, prior_is_subset=False, update_max=1e3, probabilities=None, svrg_fullgradient_interval=2, 
                 kappa_image = None, prior_in_precond = False, freeze_precond_epoch = numpy.inf, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_fun = obj_fun
        self.prior = prior
        self.save_path = save_path
        self.single_modality_update = single_modality_update
        self.prior_is_subset = prior_is_subset

        num_subsets = obj_fun.get_num_subsets()

        super(BSREMmm_of, self).__init__(initial = initial, initial_step_size = initial_step_size, num_subsets = num_subsets,
                                         relaxation_eta = relaxation_eta, stochastic = stochastic, with_replacement = with_replacement,
                                         save_images = save_images, prior_is_subset = prior_is_subset, update_max = update_max, 
                                         probabilities=probabilities,  **kwargs)
            
        if self.single_modality_update:
            self.num_subsets *= len(initial.containers)
        if self.prior_is_subset:
            self.num_subsets += 1
            
        if kappa_image is None:
            self.kappa_image = initial.clone()
            self.kappa_image = self.kappa_image.power(0)
        else:
            self.kappa_image = kappa_image
            
        self.prior_in_precond = prior_in_precond
        if self.prior_in_precond:
            self.prior_part_of_precond = self.kappa_image.power(2) * self.prior.hessian(initial)
        self.freeze_precond_epoch = freeze_precond_epoch
        self.x_bar = None

        if svrg:
            #try:
                self.svrg = True
                if not stochastic:
                    print("SVRG requires stochastic=True. Setting to True")
                    self.stochastic = True
                self.count = 0
                self.g = initial.clone()*0
                self.sg = [initial.clone()*0 for i in range(self.num_subsets)]
                self.full_gradient()
                self.svrg_fullgradient_interval = svrg_fullgradient_interval
            #except:
            #    print("Something gone wrong with SVRG. Setting to False")
            #    self.svrg = False
        else:
            self.svrg = False
            
        if saga:
            self.saga = True
            if not stochastic:
                print("SAGA requires stochastic=True. Setting to True")
                self.stochastic = True
            self.count = 0
            self.g = initial.clone()*0
            self.sg = [initial.clone()*0 for i in range(self.num_subsets)]
            self.full_gradient()
        else:
            self.saga = False
            
        if self.svrg and self.saga:
            print("Can't be SVRG and SAGA! SVRG a coming")
            self.saga=False

        # write average_sensitivity
        if isinstance(self.average_sensitivity, BlockDataContainer):
            for i, el in enumerate(self.average_sensitivity):
                el.write(os.path.join(self.save_path, f'average_sensitivity_{i}.hv'))
        else:
            self.average_sensitivity.write(os.path.join(self.save_path, 'average_sensitivity.hv'))
            
        self.update_objective()

        print(f"Number of subsets: {self.num_subsets}")
        
    def calculate_preconditioner(self):
        """Compute the preconditioner for the current iteration."""
        # if we are using the prior in the preconditioner, we need to update the x_bar
        # Only do this once so we don't overwrite the x_bar
        if self.epoch() == self.freeze_precond_epoch and self.x_bar is None:
            self.x_bar = self.x.copy()
            self.x_bar*=0
        if self.prior_in_precond:
            if self.epoch() < self.freeze_precond_epoch:
                return ((self.x / self.average_sensitivity).power(2) + self.prior_part_of_precond).sqrt()
            else:
                return ((self.x_bar / self.average_sensitivity).power(2) + self.prior_part_of_precond).sqrt()   
        if self.epoch() < self.freeze_precond_epoch:
            return self.x / self.average_sensitivity
        else:
            return self.x_bar / self.average_sensitivity
        
    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        # note: sirf.STIR Poisson likelihood uses `get_subset_sensitivity(0) for the whole
        # sensitivity if there are no subsets in that likelihood
        return self.obj_fun.get_subset_sensitivity(subset_num)

    def subset_gradient(self, subset_num):
        ''' Compute gradient at x for a particular subset '''
        print(f"Computing subset gradient {subset_num + 1} of {self.num_subsets} subsets")

        subset_grad = self.compute_subset_gradient(subset_num, prior_gradient=None)
        return self.apply_variance_reduction_if_enabled(subset_grad, subset_num)

    def compute_prior_gradient(self):
        """Compute gradient of the prior."""
        with timing('prior gradient time'):
            return - self.kappa_image.power(2) * self.prior.gradient(self.x) # negative because we want to maximise

    def compute_subset_gradient(self, subset_num, prior_gradient):
        """Compute gradient for a specific subset, optionally using prior gradient."""
        if self.prior_is_subset and subset_num == self.num_subsets - 1:
            if prior_gradient is None:
                prior_gradient = self.compute_prior_gradient()
            self.FOV_filter(prior_gradient)
            return prior_gradient

        if self.single_modality_update:
            return self._compute_single_modality_subset_gradient(subset_num, prior_gradient)

        return self._compute_general_subset_gradient(subset_num, prior_gradient)

    def _compute_single_modality_subset_gradient(self, subset_num, prior_gradient):
        """Compute gradient for a specific modality subset."""
        modality, subset_within_modality = self.get_modality_indices(subset_num)
        modality_gradient = self.compute_modality_gradient(subset_within_modality, modality)

        if not self.prior_is_subset:
            if prior_gradient is None:
                prior_gradient = self.compute_prior_gradient()
            modality_gradient += prior_gradient / self.num_subsets

        self.FOV_filter(modality_gradient)

        return modality_gradient

    def _compute_general_subset_gradient(self, subset_num, prior_gradient):
        """Compute general gradient across all subsets."""
        general_gradient = self.compute_general_gradient(subset_num)

        if not self.prior_is_subset:
            if prior_gradient is None:
                prior_gradient = self.compute_prior_gradient()
            general_gradient += prior_gradient / self.num_subsets

        self.FOV_filter(general_gradient)

        return general_gradient

    def get_modality_indices(self, subset_num):
        modality = subset_num // self.obj_fun.get_num_subsets()
        subset_within_modality = subset_num % self.obj_fun.get_num_subsets()
        print(f"Modality {modality}, Subset {subset_within_modality}")
        return modality, subset_within_modality

    def compute_modality_gradient(self, subset_within_modality, modality):
        with timing('subset gradient time'):
            o_grad = self.obj_fun.get_single_subset_gradient(self.x, subset_within_modality, modality)
        return o_grad

    def compute_general_gradient(self, subset_num):
        with timing('subset gradient time'):
            o_grad = self.obj_fun.get_subset_gradient(self.x, subset_num)
        return o_grad

    def apply_variance_reduction_if_enabled(self, subset_grad, subset_num):
        if self.svrg:
            self.update_svrg_gradients()
            print("Adjusting gradient using SVRG")
            return (subset_grad - self.sg[subset_num]) + self.g/self.num_subsets
        if self.saga:
            self.update_saga_gradients(subset_grad, subset_num)
            return (subset_grad - self.sg[subset_num]) + self.g/self.num_subsets
        print("Not using SVRG or SAGA")
        return subset_grad
    
    def update_saga_gradients(self, subset_grad, subset_num):
        self.g += (subset_grad - self.sg[subset_num])
        self.sg[subset_num] = subset_grad

    def update_svrg_gradients(self):
        self.count = (self.count + 1) % (self.num_subsets  * self.svrg_fullgradient_interval)
        if self.count == 0:
            self.full_gradient()

    def full_gradient(self):
        """Compute full gradient at x."""
        print('Computing full gradient')
        prior_gradient = self.compute_prior_gradient() 
        self.g*=0

        for i in range(self.num_subsets):
            self.sg[i] = self.compute_subset_gradient(i, prior_gradient)
            self.g += self.sg[i]

    def update_objective(self):
        df = self.obj_fun(self.x)
        p = -self.prior(self.x)
        self.loss.append([df+p, df, p])

        # print step size
        print(f"Step size: {self.step_size()}")
        
    def write_image(self, image, file_suffix, modality_suffix=''):
        """ Helper function to write images based on their type and modality. """
        if isinstance(image, (DataContainer, STIR.ImageData)):
            filename = os.path.join(self.save_path, f'BSREMmm_of_{self.iteration}_{file_suffix}{modality_suffix}.hv')
            image.write(filename)
            print(f'Writing image to {filename}')
        elif isinstance(image, BlockDataContainer):
            for i, el in enumerate(image.containers):
                filename = os.path.join(self.save_path, f'BSREMmm_of_{self.iteration}_{file_suffix}{modality_suffix}_{i}.hv')
                el.write(filename)
                
    def add_images(self):
        """ Method to save images from the current iteration, including gradient images if using SVRG. """
        self.write_image(self.x, 'image')
        
        # If SVRG is enabled, write the gradient images as well
        if self.svrg or self.saga:
            self.write_image(self.g, 'g')
            for i, el in enumerate(self.sg):
                self.write_image(el, f'sg_{i}')
        
