import torch
from torch import vmap
import numpy as np

from .Function import Function

def pseudo_inverse_torch(H):
    """Inverse except when element is zero."""
    return torch.where(H != 0, 1/H, torch.tensor(0.0, device=H.device, dtype=H.dtype)) 

def l1_norm_torch(x):
    return torch.sum(torch.abs(x))

def l1_norm_prox_torch(x, tau):
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0)

def l2_norm_torch(x):
    return torch.sqrt(torch.sum(x**2))

def l2_norm_prox_torch(x, tau):
    return x / torch.maximum(1, l2_norm_torch(x) / tau)

def charbonnier_torch(x, eps):
    return torch.sqrt(x**2 + eps**2) - eps

def charbonnier_grad_torch(x, eps):
    return x / torch.sqrt(x**2 + eps**2)

def fair_torch(x, eps):
    return eps * (torch.abs(x) / eps - torch.log(1 + torch.abs(x) / eps))

def fair_grad_torch(x, eps):
    return x / (eps + torch.abs(x))

def perona_malik_torch(x, eps):
    return eps/2 * (1 - torch.exp(-x**2 / eps**2))

def perona_malik_grad_torch(x, eps):
    return x * torch.exp(-x**2 / eps**2) / eps**2

def nothing_torch(x, eps=0):
    return x

def nothing_grad_torch(x, eps=0):
    return 1

def norm_torch(M, func, smoothing_func, eps):

    singularvalues = torch.linalg.svdvals(M)
    singularvalues = smoothing_func(singularvalues, eps)
    return func(singularvalues)

def norm_func_torch(X, func, tau):

    S, U, V = torch.linalg.svd(X)
    S_inv = pseudo_inverse_torch(S)
    S_func = func(S, tau)

    # Reconstruct the result matrix
    return X @ V @ torch.diag(S_inv) @ torch.diag(S_func)  @ V.T

def vectorised_norm(A, func, smoothing_func, eps=0):
    def ordered_norm(A):
        return norm_torch(A, func, smoothing_func, eps)

    # Apply vmap across the last two dimensions
    return vmap(vmap(vmap(ordered_norm, in_dims=0), in_dims=0), in_dims=0)(A)

def vectorised_norm_func(A, func, tau):

    def norm_prox_element(M):
        return norm_func_torch(M, func, tau)

    return vmap(vmap(vmap(norm_prox_element, in_dims=0), in_dims=0), in_dims=0)(A)

class GPUVectorialTotalVariation(Function):
    """ 
    GPU implementation of the vectorial total variation function.
    """
    def __init__(self, eps=0, norm = 'nuclear', smoothing_function=None, numpy_out=False):        

        """Initializes the GPUVectorialTotalVariation class.
        """        

        self.eps = eps
        self.norm = norm
        self.smoothing_function = smoothing_function
        self.numpy_out = numpy_out

    def direct(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        if self.norm == 'nuclear':
            norm_func = l1_norm_torch
        elif self.norm == 'frobenius':
            norm_func = l2_norm_torch
        else:
            raise ValueError('Norm not defined')

        if self.smoothing_function == 'fair':
            smoothing_func = fair_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbonnier_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_torch
        else:
            smoothing_func = nothing_torch

        return vectorised_norm(x, norm_func, smoothing_func, self.eps)

    def __call__(self, x):

        return torch.sum(self.direct(x)).numpy()

    def proximal(self, x, tau):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        if self.norm == 'nuclear':
            norm_func = l1_norm_prox_torch
        elif self.norm == 'frobenius':
            norm_func = l2_norm_prox_torch
        else:
            raise ValueError('Norm not defined')

        return vectorised_norm_func(x, norm_func, tau).numpy() if self.numpy_out else vectorised_norm_func(x, norm_func, tau)

    def gradient(self, x):

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)

        if self.smoothing_function == 'fair':
            smoothing_func = fair_grad_torch
        elif self.smoothing_function == 'charbonnier':
            smoothing_func = charbonnier_grad_torch
        elif self.smoothing_function == 'perona_malik':
            smoothing_func = perona_malik_grad_torch
        else:
            raise ValueError('Smoothing function not defined')

        return vectorised_norm_func(x, smoothing_func, self.eps).numpy() if self.numpy_out else vectorised_norm_func(x, smoothing_func, self.eps)
