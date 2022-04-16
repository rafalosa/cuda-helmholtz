//
// Created by rafal on 25.03.2022.
//

#ifndef HELMHOLTZCUDA_HELMHOLTZSET_CUH
#define HELMHOLTZCUDA_HELMHOLTZSET_CUH

#include "Utils.cuh"
#include "CudaAllocatorInterfaces.cuh"

class HelmholtzSet{

private:

    using t_plane = SimulatorUtils::Geometry::Plane;
    using angleRad_f = float;

    unsigned int m_turns{0};
    unsigned int m_coilPolygonSides{0};

    float m_wireGauge{0.0}; // [m]
    float m_coilCenterDistance{0.0}; // [m]
    float m_dl{0.0}; // [m]
    float m_sideLen{0.0}; // [m]

    t_plane m_coilsPlane{t_plane::OO}; // Default

    angleRad_f m_externalAngleIncrement{0.0}; // [rad]
    angleRad_f m_internalAngleIncrement{0.0}; // [rad]

    float3 m_coilCenters{0.0,0.0,0.0}; // [m]
    float3 m_coilStartPoint{0.0, 0.0, 0.0}; // [m]
    float3 m_dlVec{0.0, 0.0, 0.0}; // [m]
    float3 m_dirVecs[2] = {{0.0,0.0,0.0},{0.0,0.0,0.0}};

public:
    __host__ __device__ static float3 biotSavart(const float3& dl, const float3& r, const float& I);
    __host__ __device__ HelmholtzSet(unsigned int shape,
                 float sideLength,
                 float wireGauge,
                 float coilsDistance,
                 t_plane plane,
                 unsigned int turns,
                 float dl);

    __host__ __device__ float3  pointInductionVector(const float3& point, const float& I) const;

};

#endif //HELMHOLTZCUDA_HELMHOLTZSET_CUH
