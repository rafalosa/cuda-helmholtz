//
// Created by rafal on 27.03.2022.
//

#ifndef HELMHOLTZCUDA_CUDA_TYPES_OVERLOADS_CUH
#define HELMHOLTZCUDA_CUDA_TYPES_OVERLOADS_CUH

#include "vector_types.h"
#include "vector_functions.h"

__host__ __device__ float3 operator+(const float3& v1, const float3& v2){

    return make_float3(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);

}

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



#endif //HELMHOLTZCUDA_CUDA_TYPES_OVERLOADS_CUH
