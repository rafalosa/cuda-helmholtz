//
// Created by rafal on 25.03.2022.
//

#if !defined(HELMHOLTZCUDA_HELMHOLTZSET_CUH)
#define HELMHOLTZCUDA_HELMHOLTZSET_CUH

#include "Utils.cuh"

// todo: Add the same cuda allocation interface as for Meshes.

class HelmholtzSet{

private:

    using t_plane = SimulatorUtils::Geometry::Plane;

    unsigned int _turns{0};
    float _wireGauge{0.0};
    float _coilCenterDistance{0.0};
    t_plane _coilsPlane{t_plane::OO}; // Default
    unsigned int _coilPolygonSides{0};
    float _sideLen{0.0};
    float _externalAngleIncrement_rad{0.0};
    float _internalAngleIncrement_rad{0.0};
    float _dl{0.0};

    __host__ __device__ static float3 _biotSavart(float3 dl, float3 r, float I);

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
