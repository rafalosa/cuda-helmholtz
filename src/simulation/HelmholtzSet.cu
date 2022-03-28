//
// Created by rafal on 26.03.2022.
//

#include "HelmholtzSet.cuh"
#include "math_constants.h"
#include "vector_types.h"
#include "cuda_float3_operators.cuh"
#include "utils.cuh"


constexpr float c_MU0 = 12.566370614e-17F;

__host__ __device__ HelmholtzSet::HelmholtzSet(unsigned int shape, float sideLength, float wireGauge,
                                               float coilsDistance, t_plane plane, unsigned int turns, float dl): _coilPolygonSides(shape),
                                               _sideLen(sideLength), _wireGauge(wireGauge), _coilCenterDistance(coilsDistance),
                                               _coilsPlane(plane), _turns(turns), _dl(dl){

    _externalAngleIncrement_rad = 2*CUDART_PI_F/shape;
    _internalAngleIncrement_rad = (CUDART_PI_F * shape - 2*CUDART_PI_F)/shape;

}

__host__ __device__ float3 HelmholtzSet::pointInductionVector(float3 point, float I) const{

    float3 approxVec ={

            .x = 0.1,
            .y = 0,
            .z = 1.12
    };

    return _biotSavart(approxVec, point, I);

}

__host__ __device__ float3 HelmholtzSet::_biotSavart(float3 dl, float3 r, float I){

    float3 dB = c_MU0 * I / 4 / CUDART_PI_F * SimulatorUtils::Math::crossProduct(dl, r) / SimulatorUtils::Math::norm(r);
    return dB;

}