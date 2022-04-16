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
    float m_wireGauge{0.0};
    float m_coilCenterDistance{0.0};
    t_plane m_coilsPlane{t_plane::OO}; // Default
    unsigned int m_coilPolygonSides{0};
    float m_sideLen{0.0};
    angleRad_f m_externalAngleIncrement{0.0};
    angleRad_f m_internalAngleIncrement{0.0};
    float m_dl{0.0};
    float3 m_coilCenters{0,0,0};
    float3 m_coilStartPoint{0, 0, 0};
    float3 m_dlVec{0, 0, 0};
    float3 m_dirVecs[2] = {{0,0,0},{0,0,0}};

    __host__ __device__ static float3 biotSavart(float3 dl, float3 r, float I);

public:

    __host__ __device__ HelmholtzSet(unsigned int shape,
                 float sideLength,
                 float wireGauge,
                 float coilsDistance,
                 t_plane plane,
                 unsigned int turns,
                 float dl);

    __host__ __device__ float3  pointInductionVector(float3 point, float I) const;

};

#endif //HELMHOLTZCUDA_HELMHOLTZSET_CUH
