#ifndef HELMHOLTZCUDA_CUDAMACROS_CUH
#define HELMHOLTZCUDA_CUDAMACROS_CUH

#define CUDA_ERRCHK(expression) \
if(expression != cudaSuccess){ \
    throw std::runtime_error("Cuda error"); } \

#endif //HELMHOLTZCUDA_CUDAMACROS_CUH
