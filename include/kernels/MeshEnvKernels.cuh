#ifndef HELMHOLTZCUDA_MESHENVKERNELS_CUH
#define HELMHOLTZCUDA_MESHENVKERNELS_CUH

#include "MeshUtils.h"

template<MeshUtils::Dim, int N>
class Mesh;

// todo: These new/delete kernels could be templated for different dimensions of meshes. Then just implement template
//  specializations. This would be cleaner. example: cudaInstantiate3DMesh would be cudaInstantiateMesh<MeshUtils::Dim::D3>

template<int N>
__global__ void cudaInstantiate3DMesh(Mesh<MeshUtils::Dim::D3, N>** meshPtr,
                                      MeshUtils::Units meshUnits,
                                      float x1Boundary, float x2Boundary,
                                      float y1Boundary, float y2Boundary,
                                      float z1Boundary, float z2Boundary){

    *(meshPtr) = new Mesh<MeshUtils::Dim::D3, N>(meshUnits, x1Boundary,
                                                 x2Boundary, y1Boundary,
                                                 y2Boundary, z1Boundary,
                                                 z2Boundary);
}

template<int N> // For testing.
__global__ void getPointGPUMesh3D(Mesh<MeshUtils::Dim::D3, N>** meshPtr, size_t x, size_t y, size_t z, float3* pointPtr){

    *(pointPtr) = (*(meshPtr)) -> get(x, y, z);
}

template<int N>
__global__ void cudaDelete3DMesh(Mesh<MeshUtils::Dim::D3, N>** meshPtr){

    delete *(meshPtr);
}

#endif //HELMHOLTZCUDA_MESHENVKERNELS_CUH
