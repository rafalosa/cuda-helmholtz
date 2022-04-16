#include <iostream>
#include <memory>

#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "HelmholtzEnvKernels.cuh"
#include "HelmholtzSet.cuh"
#include "Utils.cuh"
#include "cuda_float3_operators.cuh"
#include "vector_types.h"
#include "CudaMacros.cuh"
#include "CudaAllocatorInterfaces.cuh"

using namespace SimulatorUtils::Structures;
using namespace CUDAUtils::Memory;

int main() {

    int count;
    CUDA_ERRCHK(cudaGetDeviceCount(&count)) // Getting CUDA devices count.

    std::cout << "Cuda devices available: " << count << std::endl;

    CUDAUtils::showCudaDeviceProps(0); // Displaying properties of 0th device.

    auto ent = std::make_unique<HelmholtzSet>(4, 6, 1.9, 100, SimulatorUtils::Geometry::Plane::XY, 1, 1); // Host memory allocation for HelmholtzSet object.

    auto res = ent->pointInductionVector(make_float3(0, 0, 0), 10); // CPU aka Host compute.

    std::cout <<"CPU result: " << res << std::endl;  // Displaying CPU result.

    float3* resultGPU; // Host pointer to GPU memory storing the result.
    CUDA_ERRCHK(cudaMalloc((void**)&resultGPU, sizeof(float3))) // Allocating GPU memory for result.

    auto entGPU = newCudaInstance<HelmholtzSet>(4, 6, 1.9, 100, SimulatorUtils::Geometry::Plane::XY, 1, 1);

    GPUComputeTest<<<1,1>>>(entGPU, make_float3(0, 0, 0),10,resultGPU); // GPU compute using the object created on the GPU.

    deleteCudaInstance(entGPU);

    auto resultHost = std::make_unique<float3>(); // Pointer to GPU result on Host.
    CUDA_ERRCHK(cudaMemcpy(resultHost.get(), resultGPU, sizeof(float3), cudaMemcpyDeviceToHost)) // Copying GPU result memory to Host pointer.

    cudaFree(resultGPU); // Deallocating memory on GPU.

    std::cout <<"GPU result: " << *resultHost << std::endl; // Displaying GPU result.

    return 0;
}