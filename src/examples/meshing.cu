#include <iostream>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "Mesher.cuh"
#include <memory>
#include "cuda_float3_operators.cuh"

int main() {

    int count;
    auto err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    auto msh1D = std::make_unique<Mesh<MeshUtils::Dim::D1>>(MeshUtils::Units::METERS, 100, -100, 100);

    auto msh2D = std::make_unique<Mesh<MeshUtils::Dim::D2>>(MeshUtils::Units::METERS, 100, -100, 100, -100, 100);

    auto msh3D = std::make_unique<Mesh<MeshUtils::Dim::D3>>(MeshUtils::Units::METERS, 100, -100, 100, -100, 100, -100, 100);


    auto val = (*msh3D).get(99,99,99);

    std::cout << val << std::endl;

    return 0;
}
