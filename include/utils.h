//
// Created by rafal on 27.03.2022.
//

#if !defined(HELMHOLTZCUDA_UTILS_H)
#define HELMHOLTZCUDA_UTILS_H

#include <initializer_list>
#include <iostream>

namespace CUDAUtils{

    void showCudaDeviceProps(int device);
}

namespace SimulatorUtils{
    namespace Geometry{
        enum class Plane : short{

            OO = -1,
            XY = 0,
            YX = 0,
            YZ = 1,
            ZY = 1,
            XZ = 2,
            ZX = 2
        };
    }

    namespace Structures{
        struct vec3D{ // Not used, since I've found float3 in cuda headers.
            float _vals[3] = {0};
            float& operator[](unsigned int ind);
            vec3D(std::initializer_list<float> list);
        };
    }
    namespace Math{
        __host__ __device__  float3 crossProduct(float3 v1, float3 v2);

        __host__ __device__ float norm(float3 vec);

        __host__ __device__ void assignLinearSpace(float boundary1, float boundary2, size_t steps, float step, float* target);
    }
}
#endif //HELMHOLTZCUDA_UTILS_H
