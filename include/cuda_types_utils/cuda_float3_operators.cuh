//
// Created by rafal on 27.03.2022.
//

// Forward declarations of operators for float3 cuda type.

#if !defined(HELMHOLTZCUDA_FLOAT3_OVERLOADS_CUH)
#define HELMHOLTZCUDA_FLOAT3_OVERLOADS_CUH

#include "vector_types.h"
#include "vector_functions.h"
#include <iostream>

// ------------------ Arithmetic operators ------------------

__host__ __device__ float3 operator+(const float3& v1, const float3& v2);

// Full declarations are here to avoid doing forward declarations of specialized templates.

template<class ScalarType>
__host__ __device__ float3 operator*(const float3& vec, const ScalarType& scalar){

    return make_float3(vec.x * scalar, vec.y * scalar, vec.z*scalar);

}

template<class ScalarType>
__host__ __device__ float3 operator*(const ScalarType& scalar, const float3& vec){

    return make_float3(vec.x * scalar, vec.y * scalar, vec.z*scalar);

}

template<class ScalarType>
__host__ __device__ float3 operator/(const ScalarType& scalar, const float3& vec){

    return make_float3(vec.x / scalar, vec.y / scalar, vec.z / scalar);

}

template<class ScalarType>
__host__ __device__ float3 operator/(const float3& vec, const ScalarType& scalar){

    return make_float3(vec.x / scalar, vec.y /scalar, vec.z / scalar);

}

// ------------------ Stream operators ------------------

std::ostream& operator<<(std::ostream& os, const float3& vec);

#endif //HELMHOLTZCUDA_FLOAT3_OVERLOADS_CUH
