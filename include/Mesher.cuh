//
// Created by rafal on 28.03.2022.
//

#ifndef HELMHOLTZCUDA_MESHER_CUH
#define HELMHOLTZCUDA_MESHER_CUH

#include "utils.cuh"

namespace MeshUtils{
    enum class Dim : short {
        D1 = 0,
        D2 = 1,
        D3 = 2
    };

    enum class Units : short { // Don't know if I'll use it, checking things out.

        MILLIMETERS = 1000,
        CENTIMETERS = 100,
        METERS = 1,
    };
}

class MeshBase{
protected:
    float meshUnitMultiplier;
    size_t steps_;
    explicit MeshBase(MeshUtils::Units unitMultiplier, size_t steps) : meshUnitMultiplier((float)unitMultiplier), steps_(steps){};
    virtual ~MeshBase() = default;

};

template<MeshUtils::Dim, int N>
class Mesh;

template<int N>
class Mesh<MeshUtils::Dim::D1, N> : protected MeshBase{

private:

    float1* pointsArray{nullptr};

public:
    Mesh(MeshUtils::Units meshUnits, float x1Boundary, float x2Boundary): MeshBase(meshUnits, N) {

        auto linearSpan = x2Boundary - x1Boundary;
        linearSpan *= meshUnitMultiplier;
        float step = (linearSpan / (float)steps_) * meshUnitMultiplier;

        auto linArr = new float [steps_];
        pointsArray = new float1 [steps_];

        SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps_, step, linArr);

        for(size_t i = 0; i < steps_; i++){

            pointsArray[i].x = linArr[i];
        }

        delete[] linArr;
    }
    ~Mesh() override {

        delete[] pointsArray;
    }
    static size_t size() {

        return N*sizeof(float1);
    }
    float1 get(size_t index){

        return pointsArray[index];
    }
};

template<int N>
class Mesh<MeshUtils::Dim::D2, N> : protected MeshBase{

private:
    float2** pointsArray{nullptr};

public:
    Mesh(MeshUtils::Units meshUnits, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary): MeshBase(meshUnits, N) {

        meshUnitMultiplier = (float)meshUnits;

        auto linearSpanX = x2Boundary - x1Boundary;
        linearSpanX *= meshUnitMultiplier;
        float stepX = (linearSpanX / (float)steps_) * meshUnitMultiplier;

        auto linearSpanY = y2Boundary - y1Boundary;
        linearSpanY *= meshUnitMultiplier;
        float stepY = (linearSpanX / (float)steps_) * meshUnitMultiplier;

        auto linArrX = new float [steps_];
        auto linArrY = new float [steps_];

        SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps_, stepX, linArrX);
        SimulatorUtils::Math::assignLinearSpace(y1Boundary, y2Boundary, steps_, stepY, linArrY);

        pointsArray = new float2* [steps_];

        for(size_t i = 0; i < steps_; i++){

            pointsArray[i] = new float2[steps_];
        }

        for(size_t i = 0; i < steps_; i++){
            for(size_t j = 0; j < steps_; j++){

                pointsArray[i][j].x = linArrX[i];
                pointsArray[i][j].y = linArrY[j];
            }
        }

        delete[] linArrX; delete[] linArrY;

    }
    ~Mesh() override {

        for(size_t i = 0; i < steps_; i++){

            delete[] pointsArray[i];
        }
        delete[] pointsArray;
    }
    static size_t size() {

        return N*N*sizeof(float2);

    }
    float2 get(size_t idX, size_t idY){

        return pointsArray[idX][idY];

    }
};

template<int N>
class Mesh<MeshUtils::Dim::D3, N> : protected MeshBase{

private:
    float3*** pointsArray{nullptr};

public:
    Mesh(MeshUtils::Units meshUnits, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary, float z1Boundary, float z2Boundary): MeshBase(meshUnits, N) {

        meshUnitMultiplier = (float)meshUnits;

        auto linearSpanX = x2Boundary - x1Boundary;
        linearSpanX *= meshUnitMultiplier;
        float stepX = (linearSpanX / (float)steps_) * meshUnitMultiplier;

        auto linearSpanY = y2Boundary - y1Boundary;
        linearSpanY *= meshUnitMultiplier;
        float stepY = (linearSpanX / (float)steps_) * meshUnitMultiplier;

        auto linearSpanZ = z2Boundary - z1Boundary;
        linearSpanZ *= meshUnitMultiplier;
        float stepZ = (linearSpanX / (float)steps_) * meshUnitMultiplier;

        auto linArrX = new float [steps_];
        auto linArrY = new float [steps_];
        auto linArrZ = new float [steps_];

        SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps_, stepX, linArrX);
        SimulatorUtils::Math::assignLinearSpace(y1Boundary, y2Boundary, steps_, stepY, linArrY);
        SimulatorUtils::Math::assignLinearSpace(z1Boundary, z2Boundary, steps_, stepZ, linArrZ);

        pointsArray = new float3** [steps_];

        for(size_t i = 0; i < steps_; i++){

            pointsArray[i] = new float3*[steps_];

            for(size_t j = 0; j < steps_; j++) {

                pointsArray[i][j] = new float3[steps_];
            }
        }

        for(size_t i = 0; i < steps_; i++){
            for(size_t j = 0; j < steps_; j++){
                for(size_t k = 0; k < steps_; k++) {

                    pointsArray[i][j][k].x = linArrX[i];
                    pointsArray[i][j][k].y = linArrY[j];
                    pointsArray[i][j][k].z = linArrZ[k];
                }
            }
        }

        delete[] linArrX; delete[] linArrY; delete[] linArrZ;
    }
    ~Mesh() override {

        for(size_t i = 0; i < steps_; i++){
            for(size_t j = 0; j < steps_; j++) {

                delete[] pointsArray[i][j];
            }
            delete[] pointsArray[i];
        }
        delete[] pointsArray;
    };

    static size_t size() {

        return N*N*N*sizeof(float3);

    }
    float3 get(size_t idX, size_t idY, size_t idZ){

        return pointsArray[idX][idY][idZ];

    }
};


#endif //HELMHOLTZCUDA_MESHER_CUH


