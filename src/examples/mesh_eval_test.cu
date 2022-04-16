#include <iostream>
#include <memory>

#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "HelmholtzEnvKernels.cuh"
#include "HelmholtzSet.cuh"
#include "Utils.cuh"
#include "Mesher.cuh"
#include "cuda_float3_operators.cuh"
#include "vector_types.h"
#include "CudaMacros.cuh"
#include "CudaAllocatorInterfaces.cuh"

using namespace CUDAUtils::Memory;

int main() {

//    auto GPUCoils = newCudaInstance<HelmholtzSet>(8, 60, 1.9, 200,
//                                                                     SimulatorUtils::Geometry::Plane::XY, 100, 0.1);
//    auto GPUMesh = newCudaInstance<Mesh<MeshUtils::Dim::D3, 10>>
//    (MeshUtils::Units::CENTIMETERS, -100, 100, -100, 100, -100, 100);
//
//    /// Calculations code....
//
//    deleteCudaInstance(GPUMesh);
//    deleteCudaInstance(GPUCoils);


    return 0;
}