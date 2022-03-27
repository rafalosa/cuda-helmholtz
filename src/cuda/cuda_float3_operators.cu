//
// Created by rafal on 27.03.2022.
//

#include "cuda_float3_operators.cuh"
#include "vector_types.h"
#include "vector_functions.h"
#include <iostream>

__host__ __device__ float3 operator+(const float3& v1, const float3& v2){

    return make_float3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);

}

std::ostream& operator<<(std::ostream& os, const float3& vec){

    os <<"["<< vec.x << ", " << vec.y << ", " << vec.z << "]";
    return os;
}
