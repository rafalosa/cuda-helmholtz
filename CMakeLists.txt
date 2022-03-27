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
set(SOURCES ${SOURCE_DIR}/magnetics_kernels.cu
        ${SOURCE_DIR}/main.cu
        ${SOURCE_DIR}/HelmholtzSet.cu)

add_executable(HelmholtzCUDA ${SOURCES})

set_target_properties(HelmholtzCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/arv"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

target_include_directories(HelmholtzCUDA PRIVATE ${HEADERS_DIR})

