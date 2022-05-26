#ifndef HELMHOLTZCUDA_CUDAALLOCATORINTERFACES_CUH
#define HELMHOLTZCUDA_CUDAALLOCATORINTERFACES_CUH

#include "CudaMacros.cuh"

namespace CUDAUtils
{
    /**
    * Interface providers for dynamic object allocation on device memory. Based on templated functions with
    * parameters packs and templated GPU kernels.
    * Usage example:
    *
    * auto ptr = CUDAUtils::Memory::newCudaInstance<MyClass>(MyClassConstructorArguments);
    *
    * doStuffKernelExample<<<B,T>>>(ptr, args);
    *
    * Then you have to deallocate the memory
    *
    * CUDAUtils::Memory::deleteCudaInstance(ptr);
    **/

    namespace
    {
        template<class AllocType, typename...Args>
        __global__ void cudaInstantiateObject(AllocType **ptr, Args... args)
        {

            *(ptr) = new AllocType(args...);

        }

        template<class AllocType>
        __global__ void cudaDeleteObject(AllocType **ptr)
        {

            delete *(ptr);

        }
    }
    namespace Memory
    {

        template<class AllocType, typename... Args>
        __host__ AllocType** newCudaInstance(Args &&... args)
        {

            AllocType **ptr;
            CUDA_ERRCHK(cudaMalloc((void **) &ptr, sizeof(AllocType *)))
            cudaInstantiateObject<AllocType><<<1, 1>>>(ptr, std::forward<Args>(args)...);
            return ptr;

        }

        template<class AllocType>
        __host__ void deleteCudaInstance(AllocType **ptr)
        {

            cudaDeleteObject<AllocType><<<1, 1>>>(ptr);
            cudaFree(ptr);
        }

        class cudaAllocatableObject

        {   /*
            * First interface I came up with, based on inheritance and not very flexible.
            * Requires/enables one to write custom instantiation kernels for each subclass.
            * But if custom kernels are needed, you can always make a template specialization of
            * the functions and kernels above.
            */

        protected:
            cudaAllocatableObject **selfGPUInstance{nullptr};

            __host__  __device__ cudaAllocatableObject() {};

            __host__  __device__ virtual ~cudaAllocatableObject() {};

            __host__ virtual void deleteCudaInstance() {};

            __host__ virtual void newCudaInstance() {};

            __host__ cudaAllocatableObject **getGPUBasePtr() { return selfGPUInstance; };
        };
    }
}

#endif //HELMHOLTZCUDA_CUDAALLOCATORINTERFACES_CUH
