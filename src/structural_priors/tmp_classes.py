### A big mess here - will be cleaned up soon ###
### ZoomOperator not yet available in cluster so need to use STIRZoomOperator for now ###

from .Function import Function, OperatorCompositionFunction
from .Operator import Operator
import torch

from sirf.STIR import SeparableGaussianImageFilter, ImageData
import stirextra
import stir

import numpy as np

from numba import njit, prange

import torch

###########################
###### Functions ##########
###########################

class TorchNumpyWrapper(Function):

    def __init__(self, function, shape):
        self.function = function
        self.shape = shape

    def forward(self, x):
        x_t = torch.tensor(x).reshape(self.shape)
        res = self.function(x_t)
        return res.numpy()

    def gradient(self, x):
        x_t = torch.tensor(x).reshape(self.shape)
        res = self.function.gradient(x_t)
        return res.numpy().flatten()

    def hessian(self, x):
        x_t = torch.tensor(x).reshape(self.shape)
        res = self.function.hessian(x_t)
        return res.numpy().flatten()


class NumpyWrapper(Function):

    def __init__(self, function, shape):
        self.function = function
        self.shape = shape

    def forward(self, x):
        x_np = x.reshape(self.shape)
        return self.function(x_np)
    
    def gradient(self, x):
        x_np = x.reshape(self.shape)
        grad = self.function.gradient(x_np)
        return grad.flatten()
    
    def hessian(self, x):
        x_np = x.reshape(self.shape)
        hess = self.function.hessian(x_np)
        return hess.flatten()

class SIRFNumpyWrapper(Function):

    def __init__(self, function, operator, shape, template_image, im_number=None):

        self.function = function
        self.operator = operator
        self.shape = shape
        self.template_image = template_image
        self.im_number = im_number

    def forward(self, x):

        if self.im_number is None:
            arr = x.reshape(self.shape)
        else:
            arr = x.reshape(self.shape)[..., self.im_number]
        tmp_image = self.template_image.clone()
        tmp_image.fill(arr)

        
        res =  -self.function(self.operator.direct(tmp_image))
        return res
    
    def gradient(self, x):

        if self.im_number is None:
            arr = x.reshape(self.shape)
        else:
            arr = x.reshape(self.shape)[..., self.im_number]
        tmp_image = self.template_image.clone()
        tmp_image.fill(arr)

        grad = self.function.gradient(self.operator.direct(tmp_image))
        grad = -self.operator.adjoint(grad)
        if type(grad) != np.ndarray:
            grad = grad.as_array()

        if self.im_number is None:
            res = grad
        else:
            res = np.zeros(self.shape)
            res[..., self.im_number] = grad

        return res.flatten()
    
    def hessian(self, x):
        
        if self.im_number is None:
            arr = x.reshape(self.shape)
        else:
            arr = x.reshape(self.shape)[..., self.im_number]
        tmp_image = self.template_image.clone()
        tmp_image.fill(arr)

        hess = self.function.hessian(self.operator.direct(tmp_image))
        hess = -self.operator.adjoint(hess)
        if type(hess) != np.ndarray:
            hess = hess.as_array()

        if self.im_number is None:
            res = hess
        else:
            res = np.zeros(self.shape)
            res[..., self.im_number] = hess

        return res.flatten()
    
class STIRNumpyWrapper(Function):

    def __init__(self, function, operator, shape, template_image, im_number=None):

        self.function = function
        self.operator = operator
        self.shape = shape
        self.template_image = template_image
        self.im_number = im_number

    def forward(self, x):

        if self.im_number is None:
            arr = x.reshape(self.shape)
        else:
            arr = x.reshape(self.shape)[..., self.im_number]
        tmp_image = self.template_image.get_empty_copy()
        tmp_image.fill(arr)

        return -self.function(self.operator.direct(tmp_image))
    
    def gradient(self, x):

        if self.im_number is None:
            arr = x.reshape(self.shape)
        else:
            arr = x.reshape(self.shape)[..., self.im_number]
        tmp_image = self.template_image.get_empty_copy()
        tmp_image.fill(arr)

        grad = self.function.gradient(self.operator.direct(tmp_image))
        grad = -self.operator.adjoint(grad)
        if type(grad) != np.ndarray:
            grad = grad.as_array()

        if self.im_number is None:
            res = grad
        else:
            res = np.zeros(self.shape)
            res[..., self.im_number] = grad

        return res.flatten()
    

    
class PeronaMalik(Function):

    def __init__(self, delta = 0.1):

        self.delta = delta

    def forward(self, x):

        return self.delta**2 * (np.ones_like(x) - np.exp(-x**2 / self.delta**2))
    
    def gradient(self, x):

        return 2 * x * np.exp(-x**2 / self.delta**2)
    
    def hessian(self, x):

        return 2 / self.delta**2 * np.exp(-x**2 / self.delta**2) * (self.delta**2 - 2 * x**2)

@njit(parallel=True)
def _fairl21(x, delta):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2))
                acc += delta * (norm/delta - np.log(1 + norm/delta))
    return acc

@njit(parallel=True)
def _fairl21_grad(x, delta):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2))
                res[i, j, k] = x[i, j, k] / (delta + norm)
    return res

@njit(parallel=True)
def _fairl21_hess(x, delta):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2))
                res[i, j, k] = 1 / (delta + norm)**2
    return res
    
class FairL21Norm(Function):

    def __init__(self, delta = 0.1):
        self.delta = delta

    def forward(self, x):
        return _fairl21(x.astype(np.float64), self.delta)
    
    def gradient(self, x):
        return _fairl21_grad(x.astype(np.float64), self.delta)
    
    def hessian(self, x):
        return _fairl21_hess(x.astype(np.float64), self.delta)
        
@njit(parallel=True)
def _charbonnier(x, delta):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2) + delta**2)
                acc += norm - delta
    return acc

@njit(parallel=True)
def _charbonnier_grad(x, delta):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2) + delta**2)
                res[i, j, k] = x[i, j, k] / norm
    return res

@njit(parallel=True)
def _charbonnier_hess(x, delta):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2) + delta**2)
                res[i, j, k] = delta**2 / norm**3
    return res

class CharbonnierL21Norm(Function):
        
            def __init__(self, delta = 0.1):
                self.delta = delta
        
            def forward(self, x):
                res = _charbonnier(x.astype(np.float64), self.delta)
                return res
            
            def gradient(self, x):
                return _charbonnier_grad(x.astype(np.float64), self.delta)
            
            def hessian(self, x):
                return _charbonnier_hess(x.astype(np.float64), self.delta)
            
@njit(parallel=True)
def _l21norm(x):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                acc += np.sqrt(np.sum(x[i, j, k]**2))
    return acc

@njit(parallel=True)
def _l21norm_grad(x):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2))
                if norm > 0:
                    res[i, j, k] = x[i, j, k] / norm#
    return res

@njit(parallel=True)
def _l21norm_proximal(x, step_size):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                norm = np.sqrt(np.sum(x[i, j, k]**2))
                if norm > step_size:
                    res[i, j, k] = (1 - step_size / norm) * x[i, j, k]
    return res

@njit(parallel=True)
def _l21norm_convex_conjugate(x):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                acc += np.sqrt(np.sum(x[i, j, k]**2))
    return acc

class L21Norm(Function):#

    def forward(self, x):
        res = _l21norm(x)
        return res

    def gradient(self, x):
        return _l21norm_grad(x)
    
    def proximal(self, x, step_size):
        return _l21norm_proximal(x, step_size)
    
    def convex_conjugate(self, x):
        return _l21norm_convex_conjugate(x)
    
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def _kl_call(b, x, eta):
    flat_b = b.ravel()
    flat_x = x.ravel()
    flat_eta = eta.ravel()
    out = 0
    for i in prange(flat_b.size):
        val = flat_x[i] + flat_eta[i]
        out += flat_b[i] * np.log((flat_b[i] + flat_eta[i]) / val) - flat_b[i] + val
    return out

@njit(parallel=True)
def kl_convex_conjugate(b, x, eta):
    flat_b = b.ravel()
    flat_x = x.ravel()
    flat_eta = eta.ravel()
    accumulator = 0.
    for i in prange(flat_b.size):
        y = 1 - flat_x[i]
        if y > 0:
            if flat_x[i] > 0:
                accumulator += flat_x[i] * np.log(y)
            accumulator += flat_eta[i] * flat_x[i]
    return - accumulator

@njit(parallel=True)
def _kl_grad(b, x, eta):
    flat_b = b.ravel()
    flat_x = x.ravel()
    flat_eta = eta.ravel()
    grad = np.zeros_like(flat_b)
    for i in prange(flat_b.size):
        grad[i] = 1 - (flat_b[i]) / (flat_x[i] + flat_eta[i])
    return grad.reshape(b.shape)

@njit(parallel=True)
def kl_proximal(b, x, tau, eta):
    flat_b = b.ravel()
    flat_x = x.ravel()
    flat_eta = eta.ravel()
    out = np.zeros_like(flat_x)
    for i in prange(flat_b.size):
        under_root = (flat_x[i] + flat_eta[i] - tau)**2 + 4 * tau * flat_b[i]
        out[i] = (flat_x[i] - flat_eta[i] - tau) + np.sqrt(under_root)
        out[i] /= 2
    return out.reshape(b.shape)

@njit(parallel=True)
def kl_proximal_conjugate(x, b, eta, tau):
    flat_b = b.ravel()
    flat_x = x.ravel()
    flat_eta = eta.ravel()
    out = np.zeros_like(flat_x)
    for i in prange(flat_b.size):
        z = flat_x[i] + (tau * flat_eta[i])
        out[i] = 0.5 * ((z + 1) - np.sqrt((z - 1) * (z - 1) + 4 * tau * flat_b[i]))
    return out.reshape(b.shape)


class KullbackLeibler:

    def __init__(self, b, eta=None):
        self.b = b.astype(np.float64)
        
        self.eta = eta.astype(np.float64)

        super().__init__()

    def __call__(self, x):
        return _kl_call(self.b, x.astype(np.float64), self.eta)

    def gradient(self, x):
        return _kl_grad(self.b, x.astype(np.float64), self.eta)

    def proximal(self, x, tau):
        return kl_proximal(self.b, x.astype(np.float64), tau, self.eta)

    def convex_conjugate(self, x):
        return kl_convex_conjugate(self.b, x.astype(np.float64), self.eta)
    
    def proximal_conjugate(self, x, tau):
        return kl_proximal_conjugate(self.b, x.astype(np.float64), self.eta, tau)
    

###########################
###### Operators ##########
###########################
    
class Img2ArrayOperator(Operator):

    def __init__(self, template_image):
        self.template_image = template_image

    def direct(self, x):
        return x.as_array()
    
    def adjoint(self, x):
        res =  self.template_image.clone()
        res.fill(x.reshape(self.template_image.as_array().shape))
        return res

    
class ChannelwiseOperator(Operator):

    def __init__(self, operator, channel):
        self.operator = operator
        self.channel = channel

    def direct(self, x):
        self.stored_array = x
        return self.operator(x[..., self.channel])
    
    def adjoint(self, x):
        adj = self.operator.adjoint(x)
        self.stored_array[..., self.channel] = adj
        return self.stored_array
        
    
class SmoothingOperator(Operator):
    def __init__(self, fwhm):
        self.filter = SeparableGaussianImageFilter()
        self.filter.set_fwhms(fwhm)

    def direct(self, x):
        self.filter.process(x)
        return self.filter.get_output()
        
    def adjoint(self, x):
        return self.direct(x)
    
class IdentityOperator(Operator):
    def __init__(self):
        pass

    def direct(self, x): 
        return x
        
    def adjoint(self, x):
        return x
    
class CompositionOperator(Operator):
    def __init__(self, operators:list):
        self.ops = operators
    
    def direct(self, x):
        res = x.copy()
        for op in self.ops:
            res = op.direct(res)
        return res
            
    def adjoint(self, x):
        if isinstance(x, torch.Tensor):
            res = x.clone()
        else:
            res = x.copy()
        for op in self.ops[::-1]:
            res = op.adjoint(res)
        return res
    
from stir import ZoomOptions, zoom_image, FloatVoxelsOnCartesianGrid
from stirextra import to_numpy
import os

class ZoomOperator(Operator):
    def __init__(self, ref, float, preserve = 'preserve_values', make_adjoint=True):
        
        # if ref has method zoom_image_from_template, use SIRFZoomOperator
        if hasattr(ref, 'zoom_image_from_template'):
            self.operator = SIRFZoomOperator(ref, float, preserve, make_adjoint)
        else:
            self.operator = STIRZoomOperator(ref, float, preserve, make_adjoint)
            
    def direct(self, x):
        return self.operator.direct(x)
    
    def adjoint(self, x):
        return self.operator.adjoint(x)
    
class STIRZoomOperator(Operator):
    def __init__(self, ref, float, preserve = 'preserve_values', make_adjoint=True):
        
        self.ref = ref.get_uniform_copy(1)
        self.float = float.get_uniform_copy(1)
        self.preserve = preserve # 'preserve_sum', 'preserve_values', 'preserve_projections'

        # generate random number to avoid overwriting files
        import random
        i = random.randint(0, 1000000)

        self.ref.write(f'tmp_ref{i}.hv')
        self.ref_stir = FloatVoxelsOnCartesianGrid.read_from_file(f'tmp_ref{i}.hv')
        os.remove(f'tmp_ref{i}.hv')
        self.float.write(f'tmp_float{i}.hv')
        self.float_stir = FloatVoxelsOnCartesianGrid.read_from_file(f'tmp_float{i}.hv')
        os.remove(f'tmp_float{i}.hv')

        if make_adjoint:
            self.preserve = 'preserve_values'
            print('preserve_values is used for adjoint of ZoomOperator')

        if self.preserve == 'preserve_values':
            self.preserve = ZoomOptions(1)
        elif self.preserve == 'preserve_sum':
            self.preserve = ZoomOptions(0)
        elif self.preserve == 'preserve_projections':
            self.preserve = ZoomOptions(2)

    def direct(self, x):

        self.ref_stir.fill(0)
        self.float_stir.fill(x.as_array().flat)
        zoom_image(self.ref_stir, self.float_stir, self.preserve)

        return self.ref.clone().fill(to_numpy(self.float_stir))
            
    def adjoint(self, x):

        self.ref_stir.fill(x.as_array().flat)
        self.float_stir.fill(0)
        zoom_image(self.float_stir, self.ref_stir, self.preserve)

        return self.float.clone().fill(to_numpy(self.float_stir))
    
class SIRFZoomOperator(Operator):
    def __init__(self, ref, float, preserve = 'preserve_values', make_adjoint=True):
        self.ref = ref.get_uniform_copy(0)
        self.float = float.get_uniform_copy(0)
        self.preserve = preserve # 'preserve_sum', 'preserve_values', 'preserve_projections'

        if make_adjoint:
            self.preserve = 'preserve_values'
            print('preserve_values is used for adjoint of ZoomOperator')
    
    def direct(self, x):
        self.ref.fill(0)
        return x.zoom_image_from_template(self.ref, self.preserve)
            
    def adjoint(self, x):
        self.float.fill(0)
        return x.zoom_image_from_template(self.float, self.preserve)
    
class MaskOperator(Operator):
    def __init__(self, mask):
        self.mask = mask
    
    def direct(self, x):
        return x * self.mask
            
    def adjoint(self, x):
        return x * self.mask
    
class ArrayToImageOperator(Operator):
    def __init__(self, template_image):
        self.template_image = template_image
    
    def direct(self, x):
        res = self.template_image.copy()
        res.fill(x)
        return res
            
    def adjoint(self, x):
        return x.as_array()
    
from sirf.Reg import NiftyResample

class NiftyResampleOperator(Operator):

    def __init__(self, ref, float, transform):
        self.ref = ref.get_uniform_copy(0)
        self.float = float.get_uniform_copy(0)
        self.transform = transform

        self.resampler = NiftyResample()
        self.resampler.set_reference_image(ref)
        self.resampler.set_floating_image(float)
        self.resampler.set_interpolation_type_to_linear()
        self.resampler.set_padding_value(0)
        self.resampler.add_transformation(self.transform)

    def direct(self, x):
        res = self.resampler.forward(x)
        return res
    
    def adjoint(self, x):
        res = self.resampler.backward(x)
        return res