#include <iostream>
#include <memory>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "HelmholtzSet.cuh"
#include "utils.h"
#include "cuda_float3_operators.cuh"
#include "env_kernels.cuh"
#include "vector_types.h"

using namespace SimulatorUtils::Structures;

int main() {

    int count;
    auto err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    std::cout << "Cuda devices available: " << count << std::endl;

    CUDAUtils::showCudaDeviceProps(0);

    auto ent = std::make_unique<HelmholtzSet>(8, 60, 1.9, 200, SimulatorUtils::Geometry::Plane::XY, 100, 0.1);

    auto res = ent->pointInductionVector(make_float3(1.2, 12, 0), 4);

    std::cout <<"CPU result: " << res << std::endl;

    HelmholtzSet* entOnGPU;
    float3* resultGPU;

    err = cudaMalloc((void**)&resultGPU, sizeof(float3));

    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory on a device.");
    }
    err = cudaMalloc((void**)&entOnGPU, sizeof(HelmholtzSet));

    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("Failed to allocate memory on a device.");
    }

    setupGPUHelmholtzEnv<<<1,1>>>(entOnGPU,8,60,1.9,200,SimulatorUtils::Geometry::Plane::XY,100,0.1);

    GPUComputeTest<<<1,1>>>(entOnGPU, make_float3(1.2, 12, 0),4,resultGPU);

    shutdownGPUHelmholtzEnv<<<1,1>>>(entOnGPU);

    float3 resultHost;
    err = cudaMemcpy(&resultHost, resultGPU, sizeof(float3), cudaMemcpyDeviceToHost);

    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("Failed to memcpy from device.");
    }

    cudaFree(resultGPU);
    cudaFree(entOnGPU);

    std::cout <<"GPU result: " << resultHost << std::endl;

    return 0;
}
/*
 * Pseudocode for how this is supposed to work.
 *
 * Create mesh... ->  Mesh(dS, boundaries)
 * Initially the mesh will be uniformly distributed in the given volume.
 *
 * well... other stuff.
 *
 *
 * */