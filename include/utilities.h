//
// Created by rafal on 27.03.2022.
//

#ifndef HELMHOLTZCUDA_UTILITIES_H
#define HELMHOLTZCUDA_UTILITIES_H

#include <initializer_list>
#include "vector_types.h"
#include "vector_functions.h"


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
            float& operator[](unsigned int ind){return _vals[ind];}

            vec3D(std::initializer_list<float> list){
                size_t i = 0;
                for(auto& val: list){
                    _vals[i++] = val;
                }
            }
        };
    }
    namespace Math{
        inline __host__ __device__ float3 crossProduct(float3 v1, float3 v2){
            return make_float3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
        }

        inline __host__ __device__ float norm(float3 vec){

            return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
        }
    }
}
#endif //HELMHOLTZCUDA_UTILITIES_H
