//
// Created by rafal on 27.03.2022.
//

#include "cuda_float2_operators.cuh"
#include "vector_types.h"
#include "vector_functions.h"
#include <iostream>

__host__ __device__ float2 operator+(const float2& v1, const float2& v2){

    return make_float2(v1.x+v2.x, v1.y+v2.y);

}

std::ostream& operator<<(std::ostream& os, const float2& vec){

    os <<"["<< vec.x << ", " << vec.y << "]";
    return os;
}
