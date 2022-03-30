//
// Created by rafal on 29.03.2022.
//

#ifndef HELMHOLTZCUDA_CUDA_FLOAT2_OPERATORS_CUH
#define HELMHOLTZCUDA_CUDA_FLOAT2_OPERATORS_CUH

#include "vector_types.h"
#include "vector_functions.h"
#include <iostream>

// ------------------ Arithmetic operators ------------------

__host__ __device__ float2 operator+(const float2& v1, const float2& v2);

// Full declarations are here to avoid doing forward declarations of specialized templates.

template<class ScalarType>
__host__ __device__ float2 operator*(const float2& vec, const ScalarType& scalar){

    return make_float2(vec.x * scalar, vec.y * scalar);

}

template<class ScalarType>
__host__ __device__ float2 operator*(const ScalarType& scalar, const float2& vec){

    return make_float2(vec.x * scalar, vec.y * scalar);

}

template<class ScalarType>
__host__ __device__ float2 operator/(const ScalarType& scalar, const float2& vec){

    return make_float2(vec.x / scalar, vec.y / scalar);

}

template<class ScalarType>
__host__ __device__ float2 operator/(const float2& vec, const ScalarType& scalar){

    return make_float2(vec.x / scalar, vec.y /scalar);

}

// ------------------ Stream operators ------------------

std::ostream& operator<<(std::ostream& os, const float2& vec);

#endif //HELMHOLTZCUDA_CUDA_FLOAT2_OPERATORS_CUH
