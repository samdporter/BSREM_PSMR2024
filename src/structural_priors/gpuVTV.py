import os
import sys
import subprocess

# Set the environment variable
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

from Function import Function
from jax import jit, vmap, device_get
import jax.numpy as jnp



@jit
def singular_vals(A):
    return jnp.linalg.svd(A, compute_uv=False)

@jit
def sum_singular(A, eps):
    return jnp.sum(singular_vals(A))

@jit
def sum_singular_fair(A, eps):
    s = singular_vals(A)
    return jnp.sum(eps * (s/eps - jnp.log(1 + s/eps)))

@jit
def singular(A):
    return jnp.linalg.svd(A, compute_uv=True, full_matrices=False)

@jit
def fair_gradient(x, eps):
    return x / (eps + jnp.sqrt(x**2))

@jit
def nuc_norm_gradient(x, eps):
    u, s, vt = singular(x)
    s = fair_gradient(s, eps)
    return u @ jnp.diag(s) @ vt

@jit
def nuc_norm_proximal(x, tau):
    u, s, vt = singular(x)
    s = jnp.sign(s) * jnp.maximum(jnp.abs(s) - tau, 0)
    return u @ jnp.diag(s) @ vt

@jit
def nuc_norm_convex_conjugate(x):
    u, s, vt = singular(x)
    # projection of max norm to unit ball
    return jnp.max(jnp.minimum(1, s))

class GPUVectorialTotalVariation(Function):
    def __init__(self, num_batches=3, eps=1e-8, smoothing_function=None):        

        self.num_batches = num_batches
        self.eps = eps
        self.smoothing_function = smoothing_function

    def __call__(self, x):

        # Flatten the first three dimensions
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])

        total_elements = x.shape[0]
        batch_size = int(jnp.ceil(total_elements / self.num_batches))
        result = 0

        if self.smoothing_function == 'fair':
            sum_func = sum_singular_fair
        else:
            sum_func = sum_singular

        for i in range(self.num_batches):
            print(f'Batch {i+1}/{self.num_batches}')
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_elements)
            batch = x[start_idx:end_idx]
            eps_arr = jnp.ones(batch.shape[0]) * self.eps
            result += jnp.sum(vmap(sum_func)(batch, eps_arr))

        return device_get(result)
    
    def proximal(self, x, tau):

        if self.smoothing_function != None:
            raise ValueError('Cannot use proximal operator with smoothing function')
        
        # Flatten the first three dimensions
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])

        total_elements = x.shape[0]
        batch_size = int(jnp.ceil(total_elements / self.num_batches))
        result = []

        for i in range(self.num_batches):
            print(f'Batch {i+1}/{self.num_batches}')
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_elements)
            batch = x[start_idx:end_idx]
            tau_arr = jnp.ones(batch.shape[0]) * tau
            result.append(vmap(nuc_norm_proximal)(batch, tau_arr))

        return jnp.concatenate(result, axis=0).reshape(original_shape)
    
    def gradient(self, x):

        if self.smoothing_function == None:
            raise ValueError('Smoothing function not defined')

        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])

        total_elements = x.shape[0]
        batch_size = int(jnp.ceil(total_elements / self.num_batches))
        result = []

        for i in range(self.num_batches):
            print(f'Batch {i+1}/{self.num_batches}')
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_elements)
            batch = x[start_idx:end_idx]
            eps_arr = jnp.ones(batch.shape[0]) * self.eps
            result.append(vmap(nuc_norm_gradient)(batch, eps_arr))
        
        return jnp.concatenate(result, axis=0).reshape(original_shape)
    
    def convex_conjugate(self, x):

        if self.smoothing_function != None:
            raise ValueError('Cannot use convex conjugate with smoothing function')

        original_shape = x.shape
        x = x.reshape(-1, original_shape[-2], original_shape[-1])

        total_elements = x.shape[0]
        batch_size = int(jnp.ceil(total_elements / self.num_batches))
        result = 0

        for i in range(self.num_batches):
            print(f'Batch {i+1}/{self.num_batches}')
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_elements)
            batch = x[start_idx:end_idx]
            result += jnp.sum(vmap(nuc_norm_convex_conjugate)(batch))

        return device_get(result)



