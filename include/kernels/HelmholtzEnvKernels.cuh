//
// Created by rafal on 27.03.2022.
//

#ifndef HELMHOLTZCUDA_HELMHOLTZENVKERNELS_CUH
#define HELMHOLTZCUDA_HELMHOLTZENVKERNELS_CUH

#include "HelmholtzSet.cuh"

// todo: Make parameters object for helmholtz set.
// todo: These kernels should be members of the HelmholtzSet class.

__global__ void setupGPUHelmholtzEnv(HelmholtzSet* helmholtzPtr,
                                     unsigned int shape,
                                     float sideLength,
                                     float wireGauge,
                                     float coilsDistance,
                                     SimulatorUtils::Geometry::Plane plane,
                                     unsigned int turns,
                                     float dl){

    helmholtzPtr = new HelmholtzSet(shape, sideLength, wireGauge, coilsDistance, plane, turns, dl);

}

__global__ void shutdownGPUHelmholtzEnv(HelmholtzSet* helmholtzPtr){

    delete helmholtzPtr;
}

__global__ void GPUComputeTest(HelmholtzSet* ptr, float3 point, float current, float3* result){

    *(result) = ptr->pointInductionVector(point, current);

}

#endif //HELMHOLTZCUDA_HELMHOLTZENVKERNELS_CUH
