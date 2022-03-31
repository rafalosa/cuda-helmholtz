//
// Created by rafal on 28.03.2022.
//

#ifndef HELMHOLTZCUDA_MESHER_CUH
#define HELMHOLTZCUDA_MESHER_CUH

#include <cassert>

#include "Utils.cuh"
#include "CudaAllocatorInterfaces.cuh"
#include "CudaMacros.cuh"
#include "MeshUtils.h"
#include "MeshEnvKernels.cuh"

// todo: The meshes should have a constant point to point distance, now the provided space is equally divided into
//   N points along each axis.

// todo: Implement dynamic meshes.


using namespace CUDAUtils::Memory;

class MeshBase {
protected:
    float meshUnitMultiplier;
    size_t steps_;
    __host__ __device__ MeshBase(MeshUtils::Units unitMultiplier, size_t steps) : meshUnitMultiplier((float)unitMultiplier), steps_(steps){};
    __host__ __device__ virtual ~MeshBase() {}; // = default, results in compilation error (?).

};

template<MeshUtils::Dim, int N>
class Mesh;

template<int N>
class Mesh<MeshUtils::Dim::D1, N> : protected MeshBase, protected cudaAllocatableObject{

private:

    using meshType = Mesh<MeshUtils::Dim::D1, N>;
    float1 pointsArray[N]{0.0};
    float x1B, x2B;

public:
    __host__ __device__ Mesh(MeshUtils::Units meshUnits, float x1Boundary, float x2Boundary): MeshBase(meshUnits, N),
    cudaAllocatableObject(), x1B(x1Boundary), x2B(x2Boundary) {

        float linArr[N];

        SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps_, linArr, 1/meshUnitMultiplier);

        for(size_t i = 0; i < steps_; i++){

            pointsArray[i].x = linArr[i];
        }
    }
    __host__ __device__ ~Mesh() override {}

    __host__ __device__ float1 get(size_t index){

        assert(index < N);
        return pointsArray[index];
    }


    __host__ void newCudaInstance() override{
        // Not implemented.
    }

    __host__ void deleteCudaInstance() override{
        // Not implemented.
    }

    __host__ meshType** getCudaInstancePtr(){
        // Not implemented.
        return nullptr;
    }
};

template<int N>
class Mesh<MeshUtils::Dim::D2, N> : protected MeshBase, protected cudaAllocatableObject{

private:
    using meshType = Mesh<MeshUtils::Dim::D2, N>;
    float2 pointsArray[N][N]{0.0};
    float x1B, x2B, y1B, y2B;

public:
    __host__ __device__ Mesh(MeshUtils::Units meshUnits, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary): MeshBase(meshUnits, N), cudaAllocatableObject(),
         x1B(x1Boundary), x2B(x2Boundary), y1B(y1Boundary), y2B(y2Boundary){

        float linArrX[N];
        float linArrY[N];

        SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, N, linArrX, 1/meshUnitMultiplier);
        SimulatorUtils::Math::assignLinearSpace(y1Boundary, y2Boundary, N, linArrY, 1/meshUnitMultiplier);

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < N; j++){

                pointsArray[i][j].x = linArrX[i];
                pointsArray[i][j].y = linArrY[j];
            }
        }
    }
    __host__ __device__ ~Mesh() override {}

    __host__ __device__ float2 get(size_t idX, size_t idY){

        assert(idX < N);
        assert(idY < N);
        return pointsArray[idX][idY];
    }

    __host__ void newCudaInstance() override{
        // Not implemented.
    }

    __host__ void deleteCudaInstance() override{
        // Not implemented.
    }

    __host__ meshType** getCudaInstancePtr(){
        // Not implemented.
        return nullptr;
    }
};

template<int N>
class Mesh<MeshUtils::Dim::D3, N>: protected MeshBase, protected cudaAllocatableObject{

private:
    using meshType = Mesh<MeshUtils::Dim::D3, N>;
    float3 pointsArray[N][N][N]{0.0};
    float x1B, x2B, y1B, y2B, z1B, z2B;

public:
    __host__ __device__ Mesh(MeshUtils::Units meshUnits, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary, float z1Boundary, float z2Boundary):
         x1B(x1Boundary), x2B(x2Boundary),
         y1B(y1Boundary), y2B(y2Boundary),
         z1B(z1Boundary), z2B(z2Boundary),
         MeshBase(meshUnits, N), cudaAllocatableObject(){

        float linArrX[N], linArrY[N], linArrZ[N];

        SimulatorUtils::Math::assignLinearSpace(x1B, x2B, N, linArrX, 1/meshUnitMultiplier);
        SimulatorUtils::Math::assignLinearSpace(y1B, y2B, N, linArrY, 1/meshUnitMultiplier);
        SimulatorUtils::Math::assignLinearSpace(z1B, z2B, N, linArrZ, 1/meshUnitMultiplier);

        for(size_t i = 0; i < N; i++){
            for(size_t j = 0; j < N; j++){
                for(size_t k = 0; k < N; k++) {

                    pointsArray[i][j][k].x = linArrX[i];
                    pointsArray[i][j][k].y = linArrY[j];
                    pointsArray[i][j][k].z = linArrZ[k];

                }
            }
        }
    }

    __host__ __device__ ~Mesh() override {};

    __host__ __device__ float3 get(size_t idX, size_t idY, size_t idZ){

        assert(idX < N);
        assert(idY < N);
        assert(idZ < N);
        return pointsArray[idX][idY][idZ];
    }


      //! Old interface for object allocation on cuda device. Left just as an implementation example.

    __host__ void newCudaInstance() override{

        CUDA_ERRCHK(cudaMalloc((void**)&selfGPUInstance, sizeof(meshType*)))
        cudaInstantiate3DMesh<<<1, 1>>>((meshType**) selfGPUInstance, (MeshUtils::Units) meshUnitMultiplier, x1B, x2B,
                                        y1B, y2B, z1B, z2B);
    }

    __host__ void deleteCudaInstance() override{

        cudaDelete3DMesh<<<1, 1>>>((meshType**) selfGPUInstance);
        cudaFree(selfGPUInstance);
    }

    __host__ meshType** getCudaInstancePtr(){

        return (meshType**)(this -> getGPUBasePtr());
    }
};

#endif //HELMHOLTZCUDA_MESHER_CUH