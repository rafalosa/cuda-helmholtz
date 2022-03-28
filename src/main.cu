#include <iostream>
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"
#include "Mesher.cuh"

int main() {

    int count;
    auto err = cudaGetDeviceCount(&count);
    if(err != cudaSuccess){

        std::cout << "Error: " << cudaGetErrorName(err) << std::endl;
        throw std::runtime_error("cuda error");
    }

    auto msh = Mesh<MeshUtils::Dim::D1>(MeshUtils::Units::METERS, 100, -100, 100);

    return 0;
}
