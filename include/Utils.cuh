//
// Created by rafal on 27.03.2022.
//

#if !defined(HELMHOLTZCUDA_UTILS_CUH)
#define HELMHOLTZCUDA_UTILS_CUH

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
        __host__ __device__  float3 crossProduct(const float3& v1, const float3& v2);

        __host__ __device__ float norm(const float3& vec);

       __host__ __device__ void assignLinearSpace(const float& boundary1,
                                                  const float& boundary2,
                                                  const size_t& steps,
                                                  float* target,
                                                  const float& multiplier);

        __host__ __device__ float3 rotateAroundX(const float3& vec, const float& angle);
        __host__ __device__ float3 rotateAroundY(const float3& vec, const float& angle);
        __host__ __device__ float3 rotateAroundZ(const float3& vec, const float& angle);

    }
}
#endif //HELMHOLTZCUDA_UTILS_CUH
