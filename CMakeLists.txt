cmake_minimum_required(VERSION 3.21)
project(HelmholtzCUDA CUDA)
enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(HEADERS_DIR include include/cuda_types_utils include/kernels include/macros)
set(SOURCE_DIR src)
set(MUTUAL_SRC ${SOURCE_DIR}/tools/utils.cu
        ${SOURCE_DIR}/cuda/cuda_float3_operators.cu
        ${SOURCE_DIR}/cuda/cuda_float2_operators.cu
        ${SOURCE_DIR}/cuda/cuda_float1_operators.cu
        ${SOURCE_DIR}/simulation/HelmholtzSet.cu
        ${SOURCE_DIR}/simulation/Mesher.cu)

set(SOURCES_MAIN ${MUTUAL_SRC}
        ${SOURCE_DIR}/main.cu)

set(SOURCES_COMPUTE_EXAMPLE ${MUTUAL_SRC}
        ${SOURCE_DIR}/examples/gpu_cpu_compute.cu)

set(SOURCES_MESHING_EXAMPLE ${MUTUAL_SRC}
        ${SOURCE_DIR}/examples/meshing.cu)

add_executable(HelmholtzCUDAMain ${SOURCES_MAIN})
add_executable(Example_Compute ${SOURCES_COMPUTE_EXAMPLE})
add_executable(Example_Meshing ${SOURCES_MESHING_EXAMPLE})

set_target_properties(HelmholtzCUDAMain PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arv"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set_target_properties(Example_Compute PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arv"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set_target_properties(Example_Meshing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arv"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_include_directories(HelmholtzCUDAMain PRIVATE ${HEADERS_DIR})
target_include_directories(Example_Compute PRIVATE ${HEADERS_DIR})
target_include_directories(Example_Meshing PRIVATE ${HEADERS_DIR})
