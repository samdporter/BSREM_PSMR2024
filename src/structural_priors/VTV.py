from .Function import Function

def create_vectorial_total_variation(gpu=False, eps=1e-8, weights=None, smoothing_function=None):
    if gpu:
        from .gpuVTV import GPUVectorialTotalVariation
        # Initialize and return a GPUVectorialTotalVariation instance
        print("GPU")
        return GPUVectorialTotalVariation(eps=eps, smoothing_function=smoothing_function, weights=weights, numpy_out=True)
    else:
        from .cpuVTV import CPUVectorialTotalVariation
        # Initialize and return a CPUVectorialTotalVariation instance
        print("CPU")
        return CPUVectorialTotalVariation(eps=eps, weights=weights, smoothing_function=smoothing_function)