//
// Created by rafal on 28.03.2022.
//

#include "Mesher.cuh"
#include <cmath>
#include "utils.h"

Mesh<MeshUtils::Dim::D1>::Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary) {

    meshUnitMultiplier = (float)meshUnits;

    auto linearSpan = x2Boundary - x1Boundary;
    linearSpan *= meshUnitMultiplier;
    float step = (linearSpan / (float)steps) * meshUnitMultiplier;

    pointsArray = new float [steps];

    SimulatorUtils::Math::assignLinearSpace(x1Boundary, x2Boundary, steps, step, pointsArray);

}

Mesh<MeshUtils::Dim::D1>::~Mesh() {

    delete[] pointsArray;
}

Mesh<MeshUtils::Dim::D2>::Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary, float y1Boundary, float y2Boundary) {

}

Mesh<MeshUtils::Dim::D2>::~Mesh() {

}

Mesh<MeshUtils::Dim::D3>::Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary, float y1Boundary, float y2Boundary, float z1Boundary, float z2Boundary) {

}

Mesh<MeshUtils::Dim::D3>::~Mesh() {

}


