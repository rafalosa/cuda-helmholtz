//
// Created by rafal on 27.03.2022.
//

#ifndef HELMHOLTZCUDA_HELMHOLTZENVKERNELS_CUH
#define HELMHOLTZCUDA_HELMHOLTZENVKERNELS_CUH

#include "HelmholtzSet.cuh"

__global__ void GPUComputeTest(HelmholtzSet** ptr, float3 point, float current, float3* result){


    *result = (*ptr)->pointInductionVector(point, current);
}



#endif //HELMHOLTZCUDA_HELMHOLTZENVKERNELS_CUH
