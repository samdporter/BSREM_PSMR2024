import numpy as np
from numba import jit, njit, prange

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