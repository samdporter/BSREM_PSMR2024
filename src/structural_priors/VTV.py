import sys
sys.path.insert(0,'/home/sam/working/SIRF-Contribs/src/Python/sirf/contrib/structural_priors')
from Function import Function

def create_vectorial_total_variation(gpu=False, eps=1e-8, num_batches=3, smoothing_function='fair'):
    if gpu:
        from gpuVTV import GPUVectorialTotalVariation
        # Initialize and return a GPUVectorialTotalVariation instance
        return GPUVectorialTotalVariation(eps=eps, num_batches=num_batches, smoothing_function=smoothing_function)
    else:
        from cpuVTV import CPUVectorialTotalVariation
        # Initialize and return a CPUVectorialTotalVariation instance
        return CPUVectorialTotalVariation(eps=eps, smoothing_function=smoothing_function)
    

class VectorialTotalVaration(Function):

    def __init__(self, gpu=False, eps=1e-8, num_batches=3, smoothing_function='fair', operator=None):
        self.vtv = create_vectorial_total_variation(gpu, eps, num_batches, smoothing_function)