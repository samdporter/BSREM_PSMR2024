from .BlockDataContainer import BlockDataContainer
import numpy as np
from numba import njit, prange, jit

import time
from contextlib import contextmanager

# Define the context manager
@contextmanager
def timing(label: str):
    t0 = time.time()
    yield lambda: None  # This yields a do-nothing function to satisfy the 'with' syntax
    t1 = time.time()
    print(f'{label}: {t1 - t0:.6f} seconds')

@jit(forceobj=True)
def power_iteration(operator, input_shape, num_iterations=100):
    """
    Approximate the largest singular value of an operator using power iteration.

    Args:
        operator: The operator with defined forward and adjoint methods.
        num_iterations (int): The number of iterations to refine the approximation.

    Returns:
        float: The approximated largest singular value of the operator.
    """
    
    # Start with a random input of appropriate shape
    input_data = np.random.randn(*input_shape)
    length = len(input_shape)
    
    for i in range(num_iterations):
        # Apply forward operation
        output_data = operator.forward(input_data)
        
        # Apply adjoint operation
        input_data = operator.adjoint(output_data)
        
        # Normalize the result
        if length == 3:
            norm = fast_norm_parallel_3d(input_data)
        elif length == 4:
            norm = fast_norm_parallel_4d(input_data)

        input_data /= norm

        # print iteration on and remove previous line
        print(f'Iteration {i+1}/{num_iterations}', end='\r')
    
    return norm

@njit(parallel=True)
def fast_norm_parallel_3d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                total += arr[i, j, k]**2

    return np.sqrt(total)

@njit(parallel=True)
def fast_norm_parallel_4d(arr):
    s = arr.shape
    total = 0.0
    
    for i in prange(s[0]):
        for j in prange(s[1]):
            for k in prange(s[2]):
                for l in prange(s[3]):
                    total += arr[i, j, k, l]**2

    return np.sqrt(total)


class Operator():

    def __init__():
        raise NotImplementedError

    def __call__(self, x):
        return self.direct(x)

    def forward(self, x):
        return self.direct(x)

    def backward(self, x):
        return self.adjoint(x)
    
    def adjoint(self, x):
        raise NotImplementedError
    
    def direct(self, x):
        raise NotImplementedError
    
    def calculate_norm(self, x, num_iterations=10):
        if not hasattr(self, 'norm'):
            self.norm = power_iteration(self, x, num_iterations=num_iterations)
            return self.norm
        else:
            return self.norm
        
    def get_adjoint(self):
        return AdjointOperator(self)
        
    def __mul__(self, other):
        return ScaledOperator(self, other)
    
    def __rmul__(self, other):
        return ScaledOperator(self, other)
    
    def __div__(self, other):
        return ScaledOperator(self, 1/other)
    
    def __truediv__(self, other):
        return ScaledOperator(self, 1/other)

    def __rdiv__(self, other):
        return ScaledOperator(self, other)
    
class AdjointOperator(Operator):
    
    def __init__(self, operator):
        self.operator = operator

    def direct(self, x):
        return self.operator.adjoint(x)

    def adjoint(self, x):
        return self.operator.direct(x)
    
class ScaledOperator(Operator):

    def __init__(self, operator, scale):
        self.operator = operator
        self.scale = scale

    def direct(self, x):
        return self.scale * self.operator.direct(x)

    def adjoint(self, x):
        return 1/self.scale * self.operator.adjoint(x)
    
class CompositionOperator(Operator):
    
    def __init__(self, ops):
        self.ops = ops
        
    def direct(self, x):
        res = x.copy()
        for op in self.ops:
            res = op.direct(res)
        return res
    
    def adjoint(self,x):
        res = x.copy()
        for op in self.ops[::-1]:
            res = op.adjoint(res)
        return res
    
class BlockOperator(Operator):

    def __init__(self, operators, weights=None):
        self.operators = operators
        self.weights = weights

    def direct(self, x):
        res = []
        if isinstance(x, np.ndarray):
            x = np.moveaxis(x, -1, 0)
            for op, arr in zip(self.operators,x):
                res.append(op.direct(arr))
            return np.moveaxis(np.array(res), 0, -1)
        elif isinstance(x, BlockDataContainer):
            for op, arr in zip(self.operators,x.containers):
                res.append(op.direct(arr))
            return BlockDataContainer(res)
    
    def adjoint(self, x):
        res = []
        if isinstance(x, np.ndarray):
            x = np.moveaxis(x, -1, 0)
            for op, arr in zip(self.operators,x):
                res.append(op.adjoint(arr))
            return np.moveaxis(np.array(res), 0, -1) 
        elif isinstance(x, BlockDataContainer):
            for op, arr in zip(self.operators,x.containers):
                res.append(op.adjoint(arr))
            return BlockDataContainer(res) 
        
# Class definitions to put DataContainer into Numpy array and back
class NumpyBlockDataContainer(Operator):

    def __init__(self, domain_geometry, operator):

        self.domain_geometry = domain_geometry
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