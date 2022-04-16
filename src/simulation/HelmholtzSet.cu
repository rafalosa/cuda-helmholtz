//
// Created by rafal on 26.03.2022.
//
#include "HelmholtzSet.cuh"
#include "math_constants.h"
#include "vector_types.h"
#include "cuda_float3_operators.cuh"
#include "Utils.cuh"

constexpr float c_MU0 = 12.566370614e-7F;

__host__ __device__ HelmholtzSet::HelmholtzSet(unsigned int shape, float sideLength, float wireGauge,
                                               float coilsDistance, t_plane plane, unsigned int turns, float dl): m_coilPolygonSides(shape),
                                                                                                                  m_sideLen(sideLength),
                                                                                                                  m_wireGauge(wireGauge),
                                                                                                                  m_coilCenterDistance(coilsDistance),
                                                                                                                  m_coilsPlane(plane),
                                                                                                                  m_turns(turns),
                                                                                                                  m_dl(dl){
    using namespace SimulatorUtils::Geometry;

    m_externalAngleIncrement = 2 * CUDART_PI_F / (float)shape;
    m_internalAngleIncrement = (CUDART_PI_F * (float)shape - 2 * CUDART_PI_F) / (float)shape;

    switch(m_coilsPlane){

        case Plane::XY:
            m_coilCenters = {0, 0, m_coilCenterDistance / 2};
            m_coilStartPoint = {m_sideLen / 2,
                                -m_sideLen / 2 * tan(m_internalAngleIncrement / 2),
                                0};
            m_dlVec = make_float3(cos(CUDART_PI_F - m_internalAngleIncrement),
                       sin(CUDART_PI_F - m_internalAngleIncrement),
                       0) * m_dl;
            m_dirVecs[0] = {0, 0, 1};

            break;

        case Plane::XZ:
            m_coilCenters = {0, m_coilCenterDistance / 2, 0};
            m_coilStartPoint = {m_sideLen / 2,
                                0,
                                -m_sideLen / 2 * tan(m_internalAngleIncrement / 2)};
            m_dlVec = make_float3(cos(CUDART_PI_F - m_internalAngleIncrement),
                                  0,
                                  sin(CUDART_PI_F - m_internalAngleIncrement)) * m_dl;
            m_dirVecs[0] = {0, 1, 0};
            break;

        case Plane::YZ:
            m_coilCenters = {m_coilCenterDistance / 2, 0, 0};
            m_coilStartPoint = {0,
                                m_sideLen / 2,
                                -m_sideLen / 2 * tan(m_internalAngleIncrement / 2)};
            m_dlVec = make_float3(0,
                                  cos(CUDART_PI_F - m_internalAngleIncrement),
                                  sin(CUDART_PI_F - m_internalAngleIncrement)) * m_dl;
            m_dirVecs[0] = {1, 0, 0};
            break;

        default:
            break;
    }
    m_dirVecs[1] = -1 * m_dirVecs[0];
}

__host__ __device__ float3 HelmholtzSet::pointInductionVector(const float3& point, const float& I) const{

    using namespace SimulatorUtils::Geometry;
    using namespace SimulatorUtils::Math;

    float3 netPointMagneticInduction{0.0,0.0,0.0};

    auto stepsForPolygonSide = (size_t)(m_sideLen / m_dl);
    int coilDir;

    for(size_t coilIdx = 0; coilIdx < 2; coilIdx++) {

        coilIdx == 0 ? coilDir = 1 : coilDir = -1;
        auto dlVec = m_dlVec;

        for (size_t turnIdx = 0; turnIdx < m_turns; turnIdx++) {

            auto currentPosition = m_coilStartPoint;

            for(size_t sideIdx = 0; sideIdx < m_coilPolygonSides; sideIdx++){

                for(size_t stepIdx = 0; stepIdx < stepsForPolygonSide; stepIdx++){

                    auto distanceVec = currentPosition - point + m_coilCenters * coilDir +
                            m_dirVecs[coilIdx] * ((float)turnIdx * m_wireGauge);

                    netPointMagneticInduction += biotSavart(dlVec, distanceVec, I);
                    currentPosition += dlVec;

                }

                switch(m_coilsPlane){

                    case Plane::XY:
                        dlVec = rotateAroundZ(dlVec,  m_internalAngleIncrement);
                        break;

                    case Plane::XZ:
                        dlVec = rotateAroundY(dlVec,  m_internalAngleIncrement);
                        break;

                    case Plane::YZ:
                        dlVec = rotateAroundX(dlVec,  m_internalAngleIncrement);
                        break;

                    default:
                        break;
                }
            }
        }
    }
    return netPointMagneticInduction;
}

__host__ __device__ float3 HelmholtzSet::biotSavart(const float3& dl, const float3& r, const float& I){

    float3 dB = c_MU0 * I / 4 / CUDART_PI_F * SimulatorUtils::Math::crossProduct(dl, r) /
            SimulatorUtils::Math::pow3(SimulatorUtils::Math::norm(r));
    return dB;

}
