#include <iostream>
#include <memory>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "HelmholtzSet.cuh"
#include "utils.h"
#include "cuda_float3_operators.cuh"


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

    //showFloat3(res);
    std::cout << res << std::endl;

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