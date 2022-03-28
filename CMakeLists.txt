cmake_minimum_required(VERSION 3.21)
project(HelmholtzCUDA CUDA)
enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(HEADERS_DIR include)
set(SOURCE_DIR src)
set(SOURCES_MAIN ${SOURCE_DIR}/tools/utils.cu
        ${SOURCE_DIR}/cuda/cuda_float3_operators.cu
        ${SOURCE_DIR}/simulation/HelmholtzSet.cu
        ${SOURCE_DIR}/main.cu
        ${SOURCE_DIR}/kernels/env_kernels.cu
        ${SOURCE_DIR}/simulation/Mesher.cu)

set(SOURCES_GPU_CPU_EXAMPLE ${SOURCE_DIR}/tools/utils.cu
        ${SOURCE_DIR}/cuda/cuda_float3_operators.cu
        ${SOURCE_DIR}/simulation/HelmholtzSet.cu
        ${SOURCE_DIR}/examples/gpu_cpu_compute.cu
        ${SOURCE_DIR}/kernels/env_kernels.cu
        ${SOURCE_DIR}/simulation/Mesher.cu
        ${SOURCE_DIR}/examples/gpu_cpu_compute.cu)

add_executable(HelmholtzCUDAMain ${SOURCES_MAIN})
add_executable(Example_GPUCPU ${SOURCES_GPU_CPU_EXAMPLE})

set_target_properties(HelmholtzCUDAMain PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arv"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set_target_properties(Example_GPUCPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arv"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_include_directories(HelmholtzCUDAMain PRIVATE ${HEADERS_DIR})
target_include_directories(Example_GPUCPU PRIVATE ${HEADERS_DIR})
