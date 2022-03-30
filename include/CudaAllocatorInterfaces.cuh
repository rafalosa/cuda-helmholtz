#ifndef HELMHOLTZCUDA_CUDAALLOCATORINTERFACES_CUH
#define HELMHOLTZCUDA_CUDAALLOCATORINTERFACES_CUH

#include "CudaMacros.cuh"

namespace CUDAUtils { namespace Memory {

    /// Interface providers for dynamic object allocation on device memory.

    class cudaAllocatableObject { // First interface I came up with, based on inheritance and not very flexible.
        // Requires one to write custom instantiation kernels for each subclass.

    protected:
        cudaAllocatableObject **selfGPUInstance{nullptr};

        __host__  __device__ cudaAllocatableObject() {};

        __host__  __device__ virtual ~cudaAllocatableObject() {};

        __host__ virtual void deleteCudaInstance() {};

        __host__ virtual void newCudaInstance() {};

        __host__ cudaAllocatableObject **getGPUBasePtr() { return selfGPUInstance; };
    };

    template<class AllocType, typename...Args>
    __global__ void cudaInstantiateObject(AllocType **ptr, Args... args) {

        *(ptr) = new AllocType(args...);

    }

    template<class AllocType>
    __global__ void cudaDeleteObject(AllocType **ptr) {

        delete *(ptr);

    }

// std::vector's emplace_back style interface.
    template<class AllocType, typename... Args>
    __host__ AllocType **newCudaInstance(Args &&... args) {

        AllocType **ptr;
        CUDA_ERRCHK(cudaMalloc((void **) &ptr, sizeof(AllocType *)))
        cudaInstantiateObject<AllocType><<<1, 1>>>(ptr, args...);
        return ptr;

    }

    template<class AllocType>
    __host__ void deleteCudaInstance(AllocType **ptr) {

        cudaDeleteObject<AllocType><<<1, 1>>>(ptr);
        cudaFree(ptr);
    }
} }

#endif //HELMHOLTZCUDA_CUDAALLOCATORINTERFACES_CUH
