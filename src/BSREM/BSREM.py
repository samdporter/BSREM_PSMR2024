#
# SPDX-License-Identifier: Apache-2.0
#
# Classes implementing the BSREM algorithm in sirf.STIR
#
# Authors:  Kris Thielemans
#
# Copyright 2024 University College London

import numpy
import sirf.STIR as STIR
from sirf.Utilities import examples_data_path

from cil.optimisation.algorithms import Algorithm 

class BSREMSkeleton(Algorithm):
    def __init__(self, data, initial, initial_step_size, relaxation_eta, **kwargs):
        super(BSREMSkeleton, self).__init__(**kwargs)
        self.x = initial.copy()
        self.data = data
        self.num_subsets = len(data)
        self.initial_step_size = initial_step_size
        self.relaxation_eta = relaxation_eta
        # compute small number to add to image in preconditioner
        # don't make it too small as otherwise the algorithm cannot recover from zeroes.
        self.eps = initial.max()/1e6
        self.average_sensitivity = initial.get_uniform_copy(0)
        print('Computing average sensitivity')
        for s in range(len(data)):
            self.average_sensitivity += self.subset_sensitivity(s)/self.num_subsets
        # add a small number to avoid division by zero in the preconditioner
        self.average_sensitivity += self.average_sensitivity.max()/1e6
        print('Done computing average sensitivity')
        self.subset = 0
        self.FOV_filter = STIR.TruncateToCylinderProcessor()
        self.configured = True
        # store images here (not yet using callback)
        self.images = [self.x.copy()]

        self.iteration = 0

    def subset_sensitivity(self, subset_num):
        raise NotImplementedError

    def subset_gradient(self, x, subset_num):
        raise NotImplementedError

    def epoch(self):
        return self.iteration // self.num_subsets

    def step_size(self):
        return self.initial_step_size / (1 + self.relaxation_eta * self.epoch())

    def update(self):
        g = self.subset_gradient(self.x, self.subset)
        self.x_update = (self.x + self.eps) * g / self.average_sensitivity * self.step_size()

        self.FOV_filter.apply(self.x_update)
        self.x += self.x_update
        # threshold to non-negative
        self.x.maximum(0, out=self.x)

        self.subset = (self.subset + 1) % self.num_subsets
        # store images here (not yet using callback)
        self.images.append(self.x.copy())

    # next is needed for CIL 21.3, but we don't do any calculation to save time
    def update_objective(self):
        self.loss.append(0)

class BSREM1(BSREMSkeleton):
    ''' BSREM implementation using sirf.STIR objective functions'''
    def __init__(self, data, obj_funs, initial, initial_step_size=1, relaxation_eta=0, **kwargs):
        '''
        construct Algorithm with lists of data and, objective functions, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters
        '''
        self.obj_funs = obj_funs
        super(BSREM1, self).__init__(data, initial, initial_step_size, relaxation_eta, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        self.obj_funs[subset_num].set_up(self.x)
        # note: sirf.STIR Poisson likelihood uses `get_subset_sensitivity(0) for the whole
        # sensitivity if there are no subsets in that likelihood
        return self.obj_funs[subset_num].get_subset_sensitivity(0)

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        return self.obj_funs[self.subset].gradient(x)


class BSREM2(BSREMSkeleton):
    ''' BSREM implementation using acquisition models and prior'''
    def __init__(self, data, acq_models, prior, initial, initial_step_size=1, relaxation_eta=0, **kwargs):
        '''
        construct Algorithm with lists of data and acquisition models, prior, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters.

        WARNING: This version will use the same prior in each subset without rescaling. You should
        therefore rescale the penalisation_factor of the prior before calling this function. This will
        change in the future.
        '''
        self.acq_models = acq_models
        self.prior = prior
        super(BSREM2, self).__init__(data, initial, initial_step_size, relaxation_eta, **kwargs)

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity for a particular subset'''
        self.acq_models[subset_num].set_up(self.data[subset_num], self.x)
        return self.acq_models[subset_num].backward(self.data[subset_num].get_uniform_copy(1))

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x for a particular subset'''
        f = self.acq_models[subset_num].forward(x)
        quotient = self.data[subset_num] / f
        return self.acq_models[subset_num].backward(quotient - 1) - self.prior.gradient(x)
    
    def update_objective(self): 
        df = []
        for i in range(self.num_subsets):
            Au = self.acq_models[i].forward(self.x)
            loss = (Au - self.data[i]*Au.log()).sum()
            df.append(loss)
        p = self.prior(self.x)
        self.loss.append([sum(df)+p, sum(df), p]+df)
    
class DC_Filter():

    def __init__(self):
        self.FOV_filter = STIR.TruncateToCylinderProcessor()

    def apply(self, x):
        for el in x.containers:
            self.FOV_filter.apply(el)

class BSREMmm(BSREMSkeleton):
    ''' BSREM implementation using multiple modalities'''
    def __init__(self, data, acq_model, prior, initial, initial_step_size=1, relaxation_eta=0, **kwargs):
        '''
        construct Algorithm with lists of data and acquisition models, prior, initial estimate, initial step size,
        step-size relaxation (per epoch) and optionally Algorithm parameters.

        WARNING: This version will use the same prior in each subset without rescaling. You should
        therefore rescale the penalisation_factor of the prior before calling this function. This will
        change in the future.
        '''

        from cil.framework import DataContainer, BlockDataContainer

        self.acq_model = acq_model # should be a BlockOperator containing all the acquisition models
        self.prior = prior # should be a BlockFunction containing all the priors or a single prior acting on all modalities
        super(BSREMmm, self).__init__(data, initial, initial_step_size, relaxation_eta, **kwargs)

        self.FOV_filter = DC_Filter()

    def subset_sensitivity(self, subset_num):
        ''' Compute sensitivity'''
        return self.acq_model[subset_num].backward(self.data[subset_num].get_uniform_copy(1))

    def subset_gradient(self, x, subset_num):
        ''' Compute gradient at x'''
        f = self.acq_model[subset_num].forward(x)
        quotient = self.data[subset_num] / f
        for el in quotient.containers:
            el_arr = el.as_array()
            el_arr = numpy.nan_to_num(el_arr, nan=0, posinf=0, neginf=0)
            el.fill(el_arr)
        return self.acq_model[subset_num].backward(quotient - 1) - self.prior.gradient(x)
    
    def update_objective(self):
        df = []
        for i in range(self.num_subsets):
            Au = self.acq_model[i].forward(self.x)
            Au_log = Au.copy()
            for el in Au_log.containers:
                el = el.log()
            loss = (Au - self.data[i]*Au_log).sum()
            del Au, Au_log
            df.append(loss)
        p = self.prior(self.x)
        self.loss.append([sum(df)+p, sum(df), p]+df)