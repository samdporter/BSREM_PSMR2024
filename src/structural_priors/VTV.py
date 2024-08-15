from .Function import Function

def create_vectorial_total_variation(gpu=False, eps=1e-8, smoothing_function=None, order=None):
    if gpu:
        from .gpuVTV import GPUVectorialTotalVariation
        # Initialize and return a GPUVectorialTotalVariation instance
        print("GPU")
        return GPUVectorialTotalVariation(eps=eps, smoothing_function=smoothing_function, 
                                          order=order, numpy_out=True)
    else:
        from .cpuVTV import CPUVectorialTotalVariation
        # Initialize and return a CPUVectorialTotalVariation instance
        print("CPU")
        return CPUVectorialTotalVariation(eps=eps, smoothing_function=smoothing_function)