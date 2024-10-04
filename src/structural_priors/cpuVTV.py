from numba import njit, prange

import numpy as np
from numba import jit
from .Function import Function
from numbers import Number

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_fair(x, eps):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                s  = np.abs(s)
                acc+= np.sum(eps * (s/eps - np.log(1 + s/eps)))
    return acc

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_charbonnier(x, eps):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                acc+= np.sum(np.sqrt(s**2 + eps**2) - eps)
    return acc

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_gradient_fair(x, eps):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                s = np.abs(s) / (eps + np.abs(s))
                s = np.diag(s)
                res[i,j,k] = np.dot(u, np.dot(s, vt))
    return res

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_hessian_fair(x, eps):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                s = eps / (eps + np.abs(s))**2
                s = np.diag(s)
                res[i,j,k] = np.dot(u, np.dot(s, vt))
    return res

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_gradient_charbonnier(x, eps):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                s /= np.sqrt(s**2 + eps**2)
                s = np.diag(s)
                res[i,j,k] = np.dot(u, np.dot(s, vt))
    return res

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_hessian_fair(x, eps):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                s = eps**2 / (eps**2 + s**2)**(3/2)
                s = np.diag(s)
                res[i,j,k] = np.dot(u, np.dot(s, vt))
    return res

@jit(nopython=True, parallel=True)
def cpu_nuc_norm(x):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                acc += np.sum(s)
    return acc

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_weights(x, weights):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                acc += np.sum(s) * weights[i,j,k]
    return acc

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_proximal(x, eps):
    res = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                # prox of l1 norm
                s = np.sign(s) * np.maximum(np.abs(s) - eps, 0)
                s = np.diag(s)
                res[i,j,k] = np.dot(u, np.dot(s, vt))
    return res

@jit(nopython=True, parallel=True)
def cpu_nuc_norm_convex_conjugate(x):
    acc = 0.
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                # projection of max norm to unit ball
                acc += np.max(np.minimum(1, s))
    return acc

class CPUVectorialTotalVariation(Function):

    def __init__(self, eps, smoothing_function='fair', weights = None):
        self.eps = eps
        self.smoothing_function = smoothing_function
        self.weights = 1

    def _apply_weights(self, x):
            # if self.weights is list of scalars, multiply each channel by the corresponding scalar
        if self.weights is not None:
            if isinstance(self.weights, list):
                for i, w in enumerate(self.weights):
                    x[..., i, :] *= w
            else:
                # If self.weights is a single scalar, apply it to all channels
                if self.weights != 1 and isinstance(self.weights, Number):
                    x *= self.weights
                else: 
                    print('Weights not applied')
        return x

    def __call__(self, x):
            return self.get_value(self._apply_weights(x))
    
    def get_value(self, x):
        if self.smoothing_function == 'fair':
            return cpu_nuc_norm_fair(x.astype(np.float64), self.eps)
        elif self.smoothing_function == 'charbonnier':
            return cpu_nuc_norm_charbonnier(x.astype(np.float64), self.eps)
        elif self.smoothing_function == None:
            return cpu_nuc_norm(x.astype(np.float64))
        else:
            return 0
        
    def gradient(self, x):

        if self.smoothing_function == None:
            raise ValueError('Smoothing function not defined')
        if self.smoothing_function == 'fair':
            return cpu_nuc_norm_gradient_fair(self._apply_weights(x).astype(np.float64), self.eps)
        elif self.smoothing_function == 'charbonnier':
            return cpu_nuc_norm_gradient_charbonnier(self._apply_weights(x).astype(np.float64), self.eps)
    
    def hessian(self, x):
        
        if self.smoothing_function == None:
            raise ValueError('Smoothing function not defined')
        if self.smoothing_function == 'fair':
            return cpu_nuc_norm_hessian_fair(self._apply_weights(x).astype(np.float64), self.eps)
        elif self.smoothing_function == 'charbonnier':
            return cpu_nuc_norm_hessian_fair(self._apply_weights(x).astype(np.float64), self.eps)


    def proximal(self, x, tau):      

        if self.smoothing_function != None:
            raise ValueError('Cannot use proximal operator with smoothing function')

        return cpu_nuc_norm_proximal(self._apply_weights(x).astype(np.float64), tau)
    
    def convex_conjugate(self, x):
        return 0