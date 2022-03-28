//
// Created by rafal on 28.03.2022.
//
#include <iostream>
#include <memory>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "HelmholtzSet.cuh"
#include "utils.cuh"
#include "cuda_float3_operators.cuh"
#include "env_kernels.cuh"
#include "vector_types.h"

using namespace SimulatorUtils::Structures;

int main() {

    int count;
    auto err = cudaGetDeviceCount(&count); // Getting CUDA devices count.
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    std::cout << "Cuda devices available: " << count << std::endl;

    CUDAUtils::showCudaDeviceProps(0); // Displaying properties of 0th device.

    auto ent = std::make_unique<HelmholtzSet>(8, 60, 1.9, 200, SimulatorUtils::Geometry::Plane::XY, 100, 0.1); // Host memory allocation for HelmholtzSet object.

    auto res = ent->pointInductionVector(make_float3(1.2, 12, 0), 4); // CPU aka Host compute.

    std::cout <<"CPU result: " << res << std::endl;  // Displaying CPU result.

    HelmholtzSet* entOnGPU; // Host pointer to GPU memory object.
    float3* resultGPU; // Host pointer to GPU memory storing the result.

    err = cudaMalloc((void**)&resultGPU, sizeof(float3)); // Allocating GPU memory for result.

    if(err != cudaSuccess){ // "Handling" possible errors.

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory on a device.");
    }
    err = cudaMalloc((void**)&entOnGPU, sizeof(HelmholtzSet)); // Allocating GPU memory for HelmholtzSet object.

    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory on a device.");
    }

    setupGPUHelmholtzEnv<<<1,1>>>(entOnGPU,8,60,1.9,200,SimulatorUtils::Geometry::Plane::XY,100,0.1); // Creating GPU environment for computation aka dynamically allocating the object on the GPU.

    GPUComputeTest<<<1,1>>>(entOnGPU, make_float3(1.2, 12, 0),4,resultGPU); // GPU compute using the object created on the GPU.

    shutdownGPUHelmholtzEnv<<<1,1>>>(entOnGPU); // Shutting down the GPU environment aka deallocating the created object.

    auto resultHost = std::make_unique<float3>(); // Pointer to GPU result on Host.
    err = cudaMemcpy(resultHost.get(), resultGPU, sizeof(float3), cudaMemcpyDeviceToHost); // Copying GPU result memory to Host pointer.

    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("Failed to memcpy from device.");
    }

    cudaFree(resultGPU); // Deallocating memory on GPU.
    cudaFree(entOnGPU);

    std::cout <<"GPU result: " << *resultHost << std::endl; // Displaying GPU result.

    return 0;
}