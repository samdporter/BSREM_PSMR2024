import numpy as np
from numba import njit, prange, jit
import jax.numpy as jnp
from jax import device_get
from jax import jit as jjit

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

class Jacobian(Operator):
    def __init__(self, voxel_sizes=(1,1,1), weights=[1,1,1],
                 bnd_cond = 'Neumann', method='forward', 
                 anatomical = None,
                 gpu=False) -> None:
        
        self.voxel_sizes= voxel_sizes
        self.weights = weights
        self.gpu = gpu

        self.anatomical = device_get(anatomical)
        if self.anatomical is None:
            self.grad = Gradient(voxel_sizes=self.voxel_sizes, method=method, bnd_cond=bnd_cond, gpu=gpu)
        else:
            self.grad = DirectionalGradient(self.anatomical, voxel_sizes=self.voxel_sizes, method=method, bnd_cond=bnd_cond, gpu=gpu)

    def direct(self, images):
        num_images = images.shape[-1]
        jac_list = [self.weights[idx] * self.grad.direct(images[..., idx]) for idx in range(num_images)]
        res =  np.stack(jac_list, axis=-2)
        # clear memory
        del jac_list
        return res

    def adjoint(self, jacobians):
        num_images = jacobians.shape[-2]
        adjoint_list = []
        for idx in range(num_images):
            adjoint_list.append(self.weights[idx] * self.grad.adjoint(jacobians[..., idx,:]))
        res =  np.stack(adjoint_list, axis=-1)
        return res
    
class Gradient(Operator):

    def __init__(self, voxel_sizes, method='forward', bnd_cond='Neumann', 
                 gpu=False):
        self.voxel_sizes = voxel_sizes
        self.method = method
        self.boundary_condition = bnd_cond

        if gpu:
            self.FD = GPUFiniteDifferenceOperator(self.voxel_sizes[0], direction=0, method=self.method, bnd_cond=self.boundary_condition)
        else:
            self.FD = CPUFiniteDifferenceOperator(self.voxel_sizes[0], direction=0, method=self.method, bnd_cond=self.boundary_condition)

    def direct(self, x):
        res = []
        for i in range(x.ndim):
            self.FD.direction = i
            self.FD.voxel_sizes= self.voxel_sizes[i]
            res.append(self.FD.direct(x))
        return np.stack(res, axis=-1)

    def adjoint(self, x):
        res = []
        for i in range(x.ndim-1):
            self.FD.direction = i
            self.FD.voxel_sizes= self.voxel_sizes[i]
            res.append(self.FD.adjoint(x[..., i]))
        return -sum(res)
    
class DirectionalGradient(Operator):

    def __init__(self, anatomical, voxel_sizes, gamma=1, eta=1e-6, method='forward', bnd_cond='Neumann', gpu=False) -> None:

        self.anatomical = device_get(anatomical)
        self.voxel_size = voxel_sizes
        self.gamma = gamma
        self.eta = eta
        self.method = method
        self.boundary_condition = bnd_cond

        if gpu:
            self.FD = GPUFiniteDifferenceOperator(self.voxel_size[0], direction=0, method=self.method, bnd_cond=self.boundary_condition)
            self.directional_op = gpu_directional_op
        else:
            self.FD = CPUFiniteDifferenceOperator(self.voxel_size[0], direction=0, method=self.method, bnd_cond=self.boundary_condition)
            self.directional_op = directional_op

        anat_list = []
        for i in range(self.anatomical.ndim):
            self.FD.direction = i
            self.FD.voxel_sizes = self.voxel_size[i]
            anat_list.append(self.FD.direct(self.anatomical))
        self.anatomical_grad = np.stack(anat_list, axis=-1)

    def direct(self, x):
        res = []
        for i in range(x.ndim):
            self.FD.direction = i
            self.FD.voxel_sizes = self.voxel_size[i]
            res.append(self.FD.direct(x))
        res = np.stack(res, axis=-1)
        return device_get(self.directional_op(res, self.anatomical_grad, self.gamma, self.eta))
        
    def adjoint(self, x):
        x = self.directional_op(x, self.anatomical_grad, self.gamma, self.eta)
        res = []
        for i in range(x.ndim-1):
            self.FD.direction = i
            self.FD.voxel_sizes = self.voxel_size[i]
            res.append(device_get(self.FD.adjoint(x[..., i])))
        return -sum(res)
    

    
class GPUFiniteDifferenceOperator(Operator):

    """
    JAX implementation of finite difference operator
    """
    
    def __init__(self, voxel_sizes, direction=None, method='forward', bnd_cond='Neumann', numpy=True):
        self.voxel_sizes= voxel_sizes
        self.direction = direction
        self.method = method
        self.boundary_condition = bnd_cond

        self.numpy = numpy # whether to return numpy array or jax array
        
        if self.voxel_sizes<= 0:
            raise ValueError('Need a positive voxel size')

    def get_slice(self, x, start, stop, end=None):
        tmp = [slice(None)] * x.ndim
        tmp[self.direction] = slice(start, stop, end)
        return tmp

    def direct(self, x):
        outa = jnp.zeros_like(x) if self.numpy else x

        if self.method == 'forward':
            outa = outa.at[tuple(self.get_slice(x, 1, -1))].set(jnp.subtract(x[tuple(self.get_slice(x, 2, None))], x[tuple(self.get_slice(x, 1, -1))]))
            if self.boundary_condition == 'Neumann':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.subtract(x[tuple(self.get_slice(x, 1, 2))], x[tuple(self.get_slice(x, 0, 1))]))
            elif self.boundary_condition == 'Periodic':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.subtract(x[tuple(self.get_slice(x, 1, 2))], x[tuple(self.get_slice(x, -1, None))]))
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.subtract(x[tuple(self.get_slice(x, 0, 1))], x[tuple(self.get_slice(x, -2, -1))]))
            else:
                raise ValueError('Not implemented')

        elif self.method == 'backward':
            outa = outa.at[tuple(self.get_slice(x, 1, -1))].set(jnp.subtract(x[tuple(self.get_slice(x, 1, -1))], x[tuple(self.get_slice(x, 0, -2))]))
            if self.boundary_condition == 'Neumann':
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.subtract(x[tuple(self.get_slice(x, -1, None))], x[tuple(self.get_slice(x, -2, -1))]))
            elif self.boundary_condition == 'Periodic':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.subtract(x[tuple(self.get_slice(x, -1, None))], x[tuple(self.get_slice(x, 0, 1))]))
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.subtract(x[tuple(self.get_slice(x, -1, None))], x[tuple(self.get_slice(x, -2, -1))]))
            else:
                raise ValueError('Not implemented')

        elif self.method == 'central':
            outa = outa.at[tuple(self.get_slice(x, 1, -1))].set(jnp.subtract(x[tuple(self.get_slice(x, 2, None))], x[tuple(self.get_slice(x, 0, -2))]) / 2)
            if self.boundary_condition == 'Neumann':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.add(x[tuple(self.get_slice(x, 1, 2))], x[tuple(self.get_slice(x, 0, 1))]) / 2)
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.add(x[tuple(self.get_slice(x, -1, None))], x[tuple(self.get_slice(x, -2, -1))]) / -2)
            elif self.boundary_condition == 'Periodic':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.subtract(x[tuple(self.get_slice(x, 1, 2))], x[tuple(self.get_slice(x, -1, None))]) / 2)
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.subtract(x[tuple(self.get_slice(x, 0, 1))], x[tuple(self.get_slice(x, -2, -1))]) / 2)
            else:
                raise ValueError('Not implemented')

        else:
            raise ValueError('Not implemented')

        if self.voxel_sizes!= 1.0:
            outa /= self.voxel_sizes

        del x

        return device_get(outa) if self.numpy else outa

    def adjoint(self, x):
        outa = jnp.zeros_like(x) if self.numpy else x

        if self.method == 'forward':
            outa = outa.at[tuple(self.get_slice(x, 1, -1))].set(jnp.subtract(x[tuple(self.get_slice(x, 1, -1))], x[tuple(self.get_slice(x, 0, -2))]))
            if self.boundary_condition == 'Neumann':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(x[tuple(self.get_slice(x, 0, 1))])
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(-x[tuple(self.get_slice(x, -2, -1))])
            elif self.boundary_condition == 'Periodic':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.subtract(x[tuple(self.get_slice(x, 0, 1))], x[tuple(self.get_slice(x, -1, None))]))
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.subtract(x[tuple(self.get_slice(x, -1, None))], x[tuple(self.get_slice(x, -2, -1))]))
            else:
                raise ValueError('Not implemented')

        elif self.method == 'backward':
            outa = outa.at[tuple(self.get_slice(x, 1, -1))].set(jnp.subtract(x[tuple(self.get_slice(x, 2, None))], x[tuple(self.get_slice(x, 1, -1))]))
            if self.boundary_condition == 'Neumann':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(x[tuple(self.get_slice(x, 1, 2))])
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(-x[tuple(self.get_slice(x, -1, None))])
            elif self.boundary_condition == 'Periodic':
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(jnp.subtract(x[tuple(self.get_slice(x, 1, 2))], x[tuple(self.get_slice(x, 0, 1))]))
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(jnp.subtract(x[tuple(self.get_slice(x, 0, 1))], x[tuple(self.get_slice(x, -1, None))]))
            else:
                raise ValueError('Not implemented')
        elif self.method == 'central':
            # interior points
            outa = outa.at[tuple(self.get_slice(x, 1, -1))].set(
                jnp.subtract(x[tuple(self.get_slice(x, 2, None))], 
                             x[tuple(self.get_slice(x, 0, -2))]) /2)
            if self.boundary_condition == 'Neumann':
                # left boundary
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(
                    jnp.subtract(x[tuple(self.get_slice(x, 1, 2))], 
                            x[tuple(self.get_slice(x, 0, 1))]) / 2)
                # right boundary
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(
                    jnp.subtract(x[tuple(self.get_slice(x, -1, None))], 
                            x[tuple(self.get_slice(x, -2, -1))]) / 2)
            elif self.boundary_condition == 'Periodic':
                # left boundary
                outa = outa.at[tuple(self.get_slice(x, 0, 1))].set(
                    jnp.subtract(x[tuple(self.get_slice(x, 1, 2))], 
                                 x[tuple(self.get_slice(x, -1, None))]) / 2)
                # right boundary
                outa = outa.at[tuple(self.get_slice(x, -1, None))].set(
                    jnp.subtract(x[tuple(self.get_slice(x, 0, 1))], 
                                 x[tuple(self.get_slice(x, -2, -1))]) / 2)
            
        else:
            raise ValueError('Not implemented')
        
        if self.voxel_sizes!= 1.0:
            outa /= self.voxel_sizes

        del x

        return device_get(outa) if self.numpy else outa
    

class CPUFiniteDifferenceOperator(Operator):

    """
    Numpy implementation of finite difference operator
    """
    
    def __init__(self, voxel_sizes, direction=None, method='forward', bnd_cond='Neumann'):
        self.voxel_sizes= voxel_sizes
        self.direction = direction
        self.method = method
        self.boundary_condition = bnd_cond
        
        if self.voxel_sizes<= 0:
            raise ValueError('Need a positive voxel size')

    def get_slice(self, x, start, stop, end=None):
        tmp = [slice(None)] * x.ndim
        tmp[self.direction] = slice(start, stop, end)
        return tmp
    

    def direct(self, x, out = None):
        
        outa = np.zeros_like(x) if out is None else out

        #######################################################################
        ##################### Forward differences #############################
        #######################################################################
                
        if self.method == 'forward':  
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 1,-1))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])               

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, 0,1))],
                            out = outa[tuple(self.get_slice(x, 0,1))]) 
                
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, 0,1))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, -1,None))])  
                
            else:
                raise ValueError('Not implemented')                
                
        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                

        elif self.method == 'backward':   
                                   
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 1, -1))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])              
            
            if self.boundary_condition == 'Neumann':
                    
                    # right boundary
                    np.subtract( x[tuple(self.get_slice(x, -1, None))], \
                                 x[tuple(self.get_slice(x, -2,-1))], \
                                 out = outa[tuple(self.get_slice(x, -1, None))]) 
                    
            elif self.boundary_condition == 'Periodic':
                  
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, -1,None))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 
        
        #######################################################################
        ##################### central differences ############################
        #######################################################################
        
        
        elif self.method == 'central':
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))]) 
            
            outa[tuple(self.get_slice(x, 1, -1))] /= 2.
            
            if self.boundary_condition == 'Neumann':
                            
                # left boundary
                np.subtract( x[tuple(self.get_slice(x, 1, 2))], \
                                 x[tuple(self.get_slice(x, 0,1))], \
                                 out = outa[tuple(self.get_slice(x, 0, 1))])  
                outa[tuple(self.get_slice(x, 0, 1))] /=2.
                
                # left boundary
                np.subtract( x[tuple(self.get_slice(x, -1, None))], \
                                 x[tuple(self.get_slice(x, -2,-1))], \
                                 out = outa[tuple(self.get_slice(x, -1, None))])
                outa[tuple(self.get_slice(x, -1, None))] /=2.                
                
            elif self.boundary_condition == 'Periodic':
                pass
                
               # left boundary
                np.subtract( x[tuple(self.get_slice(x, 1, 2))], \
                                 x[tuple(self.get_slice(x, -1,None))], \
                                 out = outa[tuple(self.get_slice(x, 0, 1))])                  
                outa[tuple(self.get_slice(x, 0, 1))] /= 2.
                
                
                # left boundary
                np.subtract( x[tuple(self.get_slice(x, 0, 1))], \
                                 x[tuple(self.get_slice(x, -2,-1))], \
                                 out = outa[tuple(self.get_slice(x, -1, None))]) 
                outa[tuple(self.get_slice(x, -1, None))] /= 2.

            else:
                raise ValueError('Not implemented')                 
                
        else:
                raise ValueError('Not implemented')                
        
        if self.voxel_sizes!= 1.0:
            outa /= self.voxel_sizes 

        return outa               
                 
        
    def adjoint(self, x, out=None):
        
        # Adjoint operation defined as  
                      
        outa = np.zeros_like(x) if out is None else out
            
        #######################################################################
        ##################### Forward differences #############################
        #######################################################################            
            

        if self.method == 'forward':    
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 1, -1))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])              
            
            if self.boundary_condition == 'Neumann':            

                # left boundary
                outa[tuple(self.get_slice(x, 0,1))] = x[tuple(self.get_slice(x, 0,1))]                
                
                # right boundary
                outa[tuple(self.get_slice(x, -1,None))] = - x[tuple(self.get_slice(x, -2,-1))]  
                
            elif self.boundary_condition == 'Periodic':            

                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, -1,None))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))]) 
                
            else:
                raise ValueError('Not implemented')                 

        #######################################################################
        ##################### Backward differences ############################
        #######################################################################                
                
        elif self.method == 'backward': 
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 1,-1))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))])             
            
            if self.boundary_condition == 'Neumann':             
                
                # left boundary
                outa[tuple(self.get_slice(x, 0,1))] = x[tuple(self.get_slice(x, 1,2))]                
                
                # right boundary
                outa[tuple(self.get_slice(x, -1,None))] = - x[tuple(self.get_slice(x, -1,None))] 
                
                
            elif self.boundary_condition == 'Periodic':
            
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, 0,1))],
                            out = outa[tuple(self.get_slice(x, 0,1))])  
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, -1,None))])              
                            
            else:
                raise ValueError('Not implemented')
                
                
        #######################################################################
        ##################### central differences ############################
        #######################################################################

        elif self.method == 'central':
            
            # interior nodes
            np.subtract( x[tuple(self.get_slice(x, 2, None))], \
                             x[tuple(self.get_slice(x, 0,-2))], \
                             out = outa[tuple(self.get_slice(x, 1, -1))]) 
            outa[tuple(self.get_slice(x, 1, -1))] /= 2.0
            

            if self.boundary_condition == 'Neumann':
                
                # left boundary
                np.add(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, 1,2))],
                            out = outa[tuple(self.get_slice(x, 0,1))])
                outa[tuple(self.get_slice(x, 0,1))] /= 2.0

                # right boundary
                np.add(x[tuple(self.get_slice(x, -1,None))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))])  

                outa[tuple(self.get_slice(x, -1,None))] /= -2.0               
                                                            
                
            elif self.boundary_condition == 'Periodic':
                
                # left boundary
                np.subtract(x[tuple(self.get_slice(x, 1,2))],\
                            x[tuple(self.get_slice(x, -1,None))],
                            out = outa[tuple(self.get_slice(x, 0,1))])
                outa[tuple(self.get_slice(x, 0,1))] /= 2.0
                
                # right boundary
                np.subtract(x[tuple(self.get_slice(x, 0,1))],\
                            x[tuple(self.get_slice(x, -2,-1))],
                            out = outa[tuple(self.get_slice(x, -1,None))])
                outa[tuple(self.get_slice(x, -1,None))] /= 2.0
                
                                
            else:
                raise ValueError('Not implemented') 
                                             
        else:
                raise ValueError('Not implemented')                  
                               
        #outa *= -1.
        if self.voxel_sizes!= 1.0:
            outa /= self.voxel_sizes                     
            
        return outa
    
@njit(parallel=True)
def directional_op(image_gradient, anatomical_gradient, gamma=1, eta=1e-6):
    """
    Calculate the directional operator of a 3D image optimized with Numba JIT
    image_gradient: 3D array of image gradients
    anatomical_gradient: 3D array of anatomical gradients
    """
    image_gradient = image_gradient.astype(np.float64)
    anatomical_gradient = anatomical_gradient.astype(np.float64)
    out = np.empty_like(image_gradient)
    
    D, H, W, i = anatomical_gradient.shape

    for d in prange(D):
        for h in prange(H):
            for w in prange(W):
                xi = anatomical_gradient[d, h, w] / (np.sqrt(np.sum(anatomical_gradient[d, h, w]**2)) + eta**2)
                out[d,h,w] = (image_gradient[d,h,w] - gamma * np.dot(image_gradient[d,h,w], xi) * xi)
    return out

@jjit
def gpu_directional_op(image_gradient, anatomical_gradient, gamma=1, eta=1e-6):
    """
    Calculate the directional operator of a 3D image optimized with jax JIT
    image_gradient: 3D array of image gradients
    anatomical_gradient: 3D array of anatomical gradients
    """

    xi = anatomical_gradient / (jnp.sqrt(jnp.sum(anatomical_gradient**2, axis=-1, keepdims=True)) + eta**2)
    out = image_gradient - gamma * jnp.sum(image_gradient * xi, axis=-1, keepdims=True) * xi
    return out

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

class BlockDataContainer():
    
    def __init__(self, datacontainers: list):
        
        self.containers = np.array(datacontainers)

    def __getitem__(self, key):
        return self.containers[key]
    
    def __setitem__(self, key, value):
        self.containers[key] = value

    def __len__(self):
        return len(self.containers)

    # function overloadings
    def __multiply__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x*y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x*y for y in self.containers])
        
    def __rmultiply__(self, x):
        return self.__multiply__(x)
    
    def __mul__(self, x):
        return self.__multiply__(x)
    
    def __rmul__(self, x):
        return self.__multiply__(x)
    
    def __add__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x+y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x+y for y in self.containers])
        
    def __radd__(self, x):
        return self.__add__(x)
    
    def __sub__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x-y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x-y for y in self.containers])
        
    def __rsub__(self, x):
        return self.__sub__(x)
    
    def __truediv__(self, x):
        if isinstance(x, BlockDataContainer):
            return BlockDataContainer([x/y for x,y in zip(self.containers, x.containers)])
        else:
            return BlockDataContainer([x/y for y in self.containers])
        
    def __rtruediv__(self, x):
        return self.__truediv__(x)
    
    def clone(self):
        return BlockDataContainer([x.copy() for x in self.containers])
    
    def copy(self):
        return self.clone()
    
    def allocate(self, value=0):
        for container in self.containers:
            container.fill(value)

    def get_uniform_copy(self, value=0):
        res = self.clone()
        res.allocate(value)
        return res
    
    @property
    def shape(self):
        return self.containers.shape