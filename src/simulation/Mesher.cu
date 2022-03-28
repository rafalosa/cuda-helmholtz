//
// Created by rafal on 28.03.2022.
//

#include "Mesher.cuh"
#include <cmath>
#include "utils.cuh"

size_t MeshBase::size(){
    return steps_;
}

Mesh<MeshUtils::Dim::D1>::Mesh(MeshUtils::Units meshUnits,
                               size_t steps,
                               float x1Boundary,
                               float x2Boundary): MeshBase(meshUnits, steps) {

    auto linearSpan = x2Boundary - x1Boundary;
    linearSpan *= meshUnitMultiplier;
    float step = (linearSpan / (float)steps) * meshUnitMultiplier;

    auto linArr = new float [steps];
    pointsArray = new float1 [steps];

    SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps, step, linArr);

    for(size_t i = 0; i < steps; i++){

        pointsArray[i].x = linArr[i];
    }

    delete[] linArr;
}

Mesh<MeshUtils::Dim::D1>::~Mesh() {

    delete[] pointsArray;
}

size_t Mesh<MeshUtils::Dim::D1>::size() {

    return steps_*sizeof(float1);
}

float1 Mesh<MeshUtils::Dim::D1>::get(size_t index){

    return pointsArray[index];
}

Mesh<MeshUtils::Dim::D2>::Mesh(MeshUtils::Units meshUnits,
                               size_t steps,
                               float x1Boundary,
                               float x2Boundary,
                               float y1Boundary,
                               float y2Boundary): MeshBase(meshUnits, steps) {

    meshUnitMultiplier = (float)meshUnits;
    steps_ = steps;

    auto linearSpanX = x2Boundary - x1Boundary;
    linearSpanX *= meshUnitMultiplier;
    float stepX = (linearSpanX / (float)steps) * meshUnitMultiplier;

    auto linearSpanY = y2Boundary - y1Boundary;
    linearSpanY *= meshUnitMultiplier;
    float stepY = (linearSpanX / (float)steps) * meshUnitMultiplier;

    auto linArrX = new float [steps];
    auto linArrY = new float [steps];

    SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps, stepX, linArrX);
    SimulatorUtils::Math::assignLinearSpace(y1Boundary, y2Boundary, steps, stepY, linArrY);

    pointsArray = new float2* [steps];

    for(size_t i = 0; i < steps; i++){

        pointsArray[i] = new float2[steps];
    }

    for(size_t i = 0; i < steps; i++){
        for(size_t j = 0; j < steps; j++){

            pointsArray[i][j].x = linArrX[i];
            pointsArray[i][j].y = linArrY[j];
        }
    }

    delete[] linArrX; delete[] linArrY;

}

Mesh<MeshUtils::Dim::D2>::~Mesh() {

    for(size_t i = 0; i < steps_; i++){

        delete[] pointsArray[i];
    }
    delete[] pointsArray;
}

size_t Mesh<MeshUtils::Dim::D2>::size() {

    return steps_*steps_*sizeof(float2);

}

float2 Mesh<MeshUtils::Dim::D2>::get(size_t idX, size_t idY){

    return pointsArray[idX][idY];

}

Mesh<MeshUtils::Dim::D3>::Mesh(MeshUtils::Units meshUnits,
                               size_t steps,
                               float x1Boundary,
                               float x2Boundary,
                               float y1Boundary,
                               float y2Boundary,
                               float z1Boundary,
                               float z2Boundary): MeshBase(meshUnits, steps) {

    meshUnitMultiplier = (float)meshUnits;
    steps_ = steps;

    auto linearSpanX = x2Boundary - x1Boundary;
    linearSpanX *= meshUnitMultiplier;
    float stepX = (linearSpanX / (float)steps) * meshUnitMultiplier;

    auto linearSpanY = y2Boundary - y1Boundary;
    linearSpanY *= meshUnitMultiplier;
    float stepY = (linearSpanX / (float)steps) * meshUnitMultiplier;

    auto linearSpanZ = z2Boundary - z1Boundary;
    linearSpanZ *= meshUnitMultiplier;
    float stepZ = (linearSpanX / (float)steps) * meshUnitMultiplier;

    auto linArrX = new float [steps];
    auto linArrY = new float [steps];
    auto linArrZ = new float [steps];

    SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps, stepX, linArrX);
    SimulatorUtils::Math::assignLinearSpace(y1Boundary, y2Boundary, steps, stepY, linArrY);
    SimulatorUtils::Math::assignLinearSpace(z1Boundary, z2Boundary, steps, stepZ, linArrZ);

    pointsArray = new float3** [steps];

    for(size_t i = 0; i < steps; i++){

        pointsArray[i] = new float3*[steps];

        for(size_t j = 0; j < steps; j++) {

            pointsArray[i][j] = new float3[steps];
        }
    }

    for(size_t i = 0; i < steps; i++){
        for(size_t j = 0; j < steps; j++){
            for(size_t k = 0; k < steps; k++) {

                pointsArray[i][j][k].x = linArrX[i];
                pointsArray[i][j][k].y = linArrY[j];
                pointsArray[i][j][k].z = linArrZ[k];
            }
        }
    }

    delete[] linArrX; delete[] linArrY; delete[] linArrZ;
}

Mesh<MeshUtils::Dim::D3>::~Mesh() {

    for(size_t i = 0; i < steps_; i++){
        for(size_t j = 0; j < steps_; j++) {

            delete[] pointsArray[i][j];
        }
        delete[] pointsArray[i];
    }
    delete[] pointsArray;
}

size_t Mesh<MeshUtils::Dim::D3>::size() {

    return steps_*steps_*steps_*sizeof(float3);

}

float3 Mesh<MeshUtils::Dim::D3>::get(size_t idX, size_t idY, size_t idZ){

    return pointsArray[idX][idY][idZ];

}
