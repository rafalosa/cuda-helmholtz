#ifndef HELMHOLTZCUDA_EVALSYSTEMFORMESH_CUH
#define HELMHOLTZCUDA_EVALSYSTEMFORMESH_CUH

#include "HelmholtzSet.cuh"
#include "Mesher.cuh"
#include "MeshUtils.h"

using namespace MeshUtils;

template<int N>
__global__ void EvalSystemForMesh(HelmholtzSet** system_ptr, Mesh<Dim::D3, N>** mesh_ptr, float3 result[][N][N][N]){

    auto X = threadIdx.x;
    auto Y = threadIdx.y;
    auto Z = blockIdx.x;

    auto threadCoords = (*mesh_ptr) -> get(X,Y,Z);

    auto pointInduction = (*system_ptr)->pointInductionVector(threadCoords, 5);

    (*result)[X][Y][Z] = pointInduction;

}

//template<int N>
//__global__ void EvalSystemForMesh(HelmholtzSet** system_ptr, Mesh<Dim::D2, N>** mesh_ptr){
//
//
//}
//template<int N>
//__global__ void EvalSystemForMesh(HelmholtzSet** system_ptr, Mesh<Dim::D1, N>** mesh_ptr){
//
//
//}

#endif //HELMHOLTZCUDA_EVALSYSTEMFORMESH_CUH
