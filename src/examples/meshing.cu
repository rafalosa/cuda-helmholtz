#include <iostream>
#include <memory>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "Mesher.cuh"
#include "cuda_float3_operators.cuh"
#include "vector_types.h"
#include "CudaMacros.cuh"

#define N 100

int main() {

    int count;
    auto err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    using MeshType = Mesh<MeshUtils::Dim::D3, N>;

    auto msh3D = new MeshType(MeshUtils::Units::METERS, -100, 100, -100, 100, -100, 100);

    auto val = msh3D->get(0, 18, 92);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 256*1024*1024);

    std::cout << "CPU generated coordinates: " << val << std::endl;

    float3* gpuPoint;
    CUDA_ERRCHK(cudaMalloc((void**)&gpuPoint, sizeof(float3)))

    msh3D->newCudaInstance();

    auto ptr = msh3D -> getCudaInstancePtr();

    getPointGPUMesh3D<<<1,1>>>(ptr, 0, 18, 92, gpuPoint);

    msh3D->deleteCudaInstance();

    float3 hostPoint;
    CUDA_ERRCHK(cudaMemcpy(&hostPoint, gpuPoint, sizeof(float3), cudaMemcpyDeviceToHost))

    std::cout << "The same GPU generated coordinates: " << hostPoint << std::endl;

    cudaFree(gpuPoint);

    float3* gpuPoint2;
    CUDA_ERRCHK(cudaMalloc((void**)&gpuPoint2, sizeof(float3)))

    auto test = CUDAUtils::Memory::newCudaInstance<MeshType>(MeshUtils::Units::METERS, -100, 100, -100, 100, -100, 100);

    getPointGPUMesh3D<<<1,1>>>(test, 0, 18, 92, gpuPoint2);

    float3 hostPoint2;
    CUDA_ERRCHK(cudaMemcpy(&hostPoint2, gpuPoint2, sizeof(float3), cudaMemcpyDeviceToHost))

    std::cout << "The same GPU generated coordinates #2: " << hostPoint << std::endl;

    deleteCudaInstance(test);

    delete msh3D;

    return 0;
}
