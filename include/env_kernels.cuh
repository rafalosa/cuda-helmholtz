//
// Created by rafal on 27.03.2022.
//

#ifndef HELMHOLTZCUDA_ENV_KERNELS_CUH
#define HELMHOLTZCUDA_ENV_KERNELS_CUH

#include "HelmholtzSet.cuh"

__global__ void setupGPUHelmholtzEnv(HelmholtzSet* helmholtzPtr,unsigned int shape,
                                     float sideLength,
                                     float wireGauge,
                                     float coilsDistance,
                                     SimulatorUtils::Geometry::Plane plane,
                                     unsigned int turns,
                                     float dl);

__global__ void shutdownGPUHelmholtzEnv(HelmholtzSet* helmholtzPtr);
__global__ void GPUComputeTest(HelmholtzSet* ptr, float3 point, float current, float3* result);

#endif //HELMHOLTZCUDA_ENV_KERNELS_CUH
