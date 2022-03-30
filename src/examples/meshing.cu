#include <iostream>
#include <memory>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "Mesher.cuh"
#include "cuda_float3_operators.cuh"
#include "vector_types.h"
#include "cudaMacros.cuh"

#define N 100

int main() {

    int count;
    auto err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    using meshType = Mesh<MeshUtils::Dim::D3, N>;

    auto msh3D = new meshType(MeshUtils::Units::METERS, -100, 100, -100, 100, -100, 100);

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

    delete msh3D;

    return 0;
}
