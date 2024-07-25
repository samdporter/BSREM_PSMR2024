from numba import njit, prange

import numpy as np
from numba import jit
from .Function import Function


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
def cpu_nuc_norm(x):
    acc = 0
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            for k in prange(x.shape[2]):
                u, s, vt = np.linalg.svd(x[i,j,k], full_matrices=False)
                acc += np.sum(s)
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

    def __init__(self, eps, smoothing_function='fair'):
        self.eps = eps
        self.smoothing_function = smoothing_function

    def __call__(self, x):
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
            return cpu_nuc_norm_gradient_fair(x.astype(np.float64), self.eps)
        elif self.smoothing_function == 'charbonnier':
            return cpu_nuc_norm_gradient_charbonnier(x.astype(np.float64), self.eps)
    
    
    def proximal(self, x, tau):      

        if self.smoothing_function != None:
            raise ValueError('Cannot use proximal operator with smoothing function')

        return cpu_nuc_norm_proximal(x.astype(np.float64), tau)
    
    def convex_conjugate(self, x):
        return 0