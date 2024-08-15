from cil.optimisation.functions import BlockFunction
from cil.framework import BlockDataContainer

class ScaledBlockFunction(BlockFunction):
    """ BlockFunction scaled by BlockDataContainer

    Args:
        BlockFunction (_type_): _description_
    """    
    
    def __init__(self, functions, scales):
        super(ScaledBlockFunction, self).__init__(functions)
        self.scales = scales
        
    def __call__(self, x):
        return super(ScaledBlockFunction, self).__call__(x) * self.scales

    def gradient(self, x):
        return super(ScaledBlockFunction, self).gradient(x) * self.scales

    def convex_conjugate(self, x):
        return super(ScaledBlockFunction, self).convex_conjugate(x) / self.scales

    def proximal(self, x, tau):
        return super(ScaledBlockFunction, self).proximal(x, tau / self.scales)

    def proximal_conjugate(self, x, tau):
        return super(ScaledBlockFunction, self).proximal_conjugate(x, tau * self.scales)

    def proximal_conjugate(self, x, tau):
        return super(ScaledBlockFunction, self).proximal_conjugate(x, tau * self.scales)

    def __str__(self):
        return super(ScaledBlockFunction, self).__str__()

    def __repr__(self):
        return super(ScaledBlockFunction, self).__repr__()

    def __add__(self, other):
        return super(ScaledBlockFunction, self).__add__(other)

    def __iadd__(self, other):
        return super(ScaledBlockFunction, self).__iadd__(other)

    def __sub__(self, other):
        return super(ScaledBlockFunction, self).__sub__(other)

    def __isub__(self, other):
        return super(ScaledBlockFunction, self).__isub__(other)

    def __mul__(self, other):
        return super(ScaledBlockFunction, self).__mul__(other)

    def __imul__(self, other):
        return super(ScaledBlockFunction, self).__imul__(other)

    def __truediv__(self, other):
        return super(ScaledBlockFunction, self).__truediv__(other)

    def __itruediv__(self, other):
        return super(ScaledBlockFunction, self).__itruediv__(other)

    def __neg__(self):
        return super(ScaledBlockFunction, self).__neg__()

    def __pos__(self):
        return super(ScaledBlockFunction, self).__pos__()

    def __abs__(