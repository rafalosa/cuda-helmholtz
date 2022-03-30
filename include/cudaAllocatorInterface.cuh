#ifndef HELMHOLTZCUDA_CUDAALLOCATORINTERFACE_CUH
#define HELMHOLTZCUDA_CUDAALLOCATORINTERFACE_CUH
/*
 * Interface provider for objects to be allocated on the device.
 *
 *
 *
 * */

class cudaAllocatableObject{

protected:
    cudaAllocatableObject** selfGPUInstance{nullptr};
    __host__  __device__ cudaAllocatableObject(){};
    __host__  __device__ virtual ~cudaAllocatableObject(){};
    __host__ virtual void deleteCudaInstance(){};
    __host__ virtual void newCudaInstance(){};
    __host__ cudaAllocatableObject** getGPUBasePtr(){ return selfGPUInstance;};
};

#endif //HELMHOLTZCUDA_CUDAALLOCATORINTERFACE_CUH
