#include <iostream>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "Mesher.cuh"
#include "HelmholtzSet.cuh"
#include <memory>
#include "cuda_types_utils/cuda_float3_operators.cuh"
#include "CudaAllocatorInterfaces.cuh"
#include "EvalSystemForMesh.cuh"

int main() {

    using namespace CUDAUtils;
    constexpr unsigned int size{20};

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    showCudaDeviceProps(0);
    auto maxThreads = props.maxThreadsPerBlock;

    auto coils = Memory::newCudaInstance<HelmholtzSet>(8, 60, 1.9, 200, SimulatorUtils::Geometry::Plane::XY, 100, 0.1);

    auto mesh = Memory::newCudaInstance<Mesh<MeshUtils::Dim::D3, size>>(MeshUtils::Units::CENTIMETERS,
                                                                        -100, 100,
                                                                        -100, 100,
                                                                        -100, 100);

    using vecArr3D = float3[size][size][size];

    vecArr3D* resultGPU;
    CUDA_ERRCHK(cudaMalloc((void**)&resultGPU, sizeof(float3)*size*size*size)) // Allocating GPU memory for result.


    // todo: Add a occupancy optimizer for distributing the workload with the occupancy maximization.
    dim3 threads(size,size,1);
    dim3 blocks(size,1,1);

    EvalSystemForMesh<<<blocks, threads>>>(coils, mesh, resultGPU);

    vecArr3D resultHost;
    CUDA_ERRCHK(cudaMemcpy(resultHost, resultGPU, sizeof(float3)*size*size*size, cudaMemcpyDeviceToHost))

    cudaFree(resultGPU);
    Memory::deleteCudaInstance(mesh);
    Memory::deleteCudaInstance(coils);

    std::cout << resultHost[9][9][9].z << std::endl;

    return 0;
}
