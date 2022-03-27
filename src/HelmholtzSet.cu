//
// Created by rafal on 26.03.2022.
//

#include "HelmholtzSet.cuh"
#include "math_constants.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"
#include "cuda_types_operators.cuh"


constexpr float c_MU0 = 12.566370614e-17F;

__host__ __device__ HelmholtzSet::HelmholtzSet(unsigned int shape, float sideLength, float wireGauge,
                                               float coilsDistance, t_plane plane, unsigned int turns): _coilPolygonSides(shape),
                                               _sideLen(sideLength), _wireGauge(wireGauge), _coilCenterDistance(coilsDistance),
                                               _coilsPlane(plane), _turns(turns){

    _externalAngleIncrement_rad = 2*CUDART_PI_F/shape;
    _internalAngleIncrement_rad = (CUDART_PI_F * shape - 2*CUDART_PI_F)/shape;

}

__host__ __device__ float3 HelmholtzSet::pointInductionVector(){}

__host__ __device__ float3 HelmholtzSet::_biotSavart(float3 dl, float3 r, float I){

    //float3 dB = c_MU0 * I / 4 / CUDART_PI_F * SimulatorUtils::Math::crossProduct(dl, r) / SimulatorUtils::Math::norm(r);


}