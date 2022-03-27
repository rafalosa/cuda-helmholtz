#include <iostream>
#include <random>
#include <memory>
#include "magnetics_kernels.cuh"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "utilities.h"

#define N 512

using namespace SimulatorUtils::Structures;

int main() {

//    int *d_a, *d_b, *d_c;
//    int size = N * sizeof(int);
//
//    cudaMalloc((void**)&d_a, size);
//    cudaMalloc((void**)&d_b, size);
//    cudaMalloc((void**)&d_c, size);
//
//    auto a = std::make_unique<int[]>(N);
//    random_ints(a.get(), N);
//
//    auto b = std::make_unique<int[]>(N);
//    random_ints(b.get(), N);
//
//    auto c = std::make_unique<int[]>(N);
//
//    cudaMemcpy(d_a, a.get(), size, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, b.get(), size, cudaMemcpyHostToDevice);
//
//    cudaAdd<<<N,1>>>(d_a,d_b,d_c);
//
//    cudaMemcpy(c.get(), d_c, size, cudaMemcpyDeviceToHost);
//
//    cudaFree(d_a);
//    cudaFree(d_b);
//    cudaFree(d_c);

    int count;
    auto err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    std::cout << "Cuda devices available: " << count << std::endl;

    cudaDeviceProp props{};

    cudaGetDeviceProperties(&props, 0);

    std::cout << "--- Information for device 0 ---" << std::endl;
    std::cout << "Name: " << props.name << std::endl;
    std::cout << "Compute capability: " << props.major <<"."<<props.minor << std::endl;
    std::cout << "Clock rate: " << props.clockRate << std::endl;
    std::cout << "Copy overlap: " << (props.deviceOverlap ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Kernel execution timeout: " << (props.kernelExecTimeoutEnabled ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Total global memory: " << props.totalGlobalMem << std::endl;
    std::cout << "Total const. memory: " << props.totalConstMem << std::endl;
    std::cout << "Memory pitch: " << props.memPitch << std::endl;
    std::cout << "Multiprocessor (mp) count: " << props.multiProcessorCount << std::endl;
    std::cout << "Shared memory per mp: " << props.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;


    return 0;
}
/*
 * Pseudocode for how this is supposed to work.
 *
 * Create mesh... ->  Mesh(dS, boundaries)
 * Initially the mesh will be uniformly distributed in the given volume.
 *
 *
 *
 *
 * */