//
// Created by rafal on 25.03.2022.
//

#ifndef HELMHOLTZCUDA_HELMHOLTZSET_CUH
#define HELMHOLTZCUDA_HELMHOLTZSET_CUH

#include "utilities.h"
#include "vector_types.h"


class HelmholtzSet{

private:

    using t_plane = SimulatorUtils::Geometry::Plane;
    using vec = SimulatorUtils::Structures::vec3D;

    unsigned int _turns{0};
    float _wireGauge{0.0};
    float _coilCenterDistance{0.0};
    t_plane _coilsPlane{t_plane::OO};
    unsigned int _coilPolygonSides{0};
    float _sideLen{0.0};
    float _externalAngleIncrement_rad{0.0};
    float _internalAngleIncrement_rad{0.0};

    __host__ __device__ static float3 _biotSavart(float3 dl, float3 r, float I);


public:

    __host__ __device__ HelmholtzSet(unsigned int shape,
                 float sideLength,
                 float wireGauge,
                 float coilsDistance,
                 t_plane plane,
                 unsigned int turns);

    __host__ __device__ float3  pointInductionVector();


};



#endif //HELMHOLTZCUDA_HELMHOLTZSET_CUH
