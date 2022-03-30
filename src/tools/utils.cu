//
// Created by rafal on 27.03.2022.
//

#include "utils.cuh"

namespace CUDAUtils{

    __host__ void showCudaDeviceProps(int device){

        cudaDeviceProp props{};

        cudaGetDeviceProperties(&props, device);

        std::cout << "--- Information for device "<< device <<" ---" << std::endl;
        std::cout << "Name: " << props.name << std::endl;
        std::cout << "Compute capability: " << props.major <<"."<<props.minor << std::endl;
        std::cout << "Clock rate: " << props.clockRate << std::endl;
        std::cout << "Copy overlap: " << (props.deviceOverlap ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Kernel execution timeout: " << (props.kernelExecTimeoutEnabled ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Total global memory: " << props.totalGlobalMem << std::endl;
        std::cout << "Total const. memory: " << props.totalConstMem << std::endl;
        std::cout << "Memory pitch: " << props.memPitch << std::endl;
        std::cout << "Multiprocessor (mp) count: " << props.multiProcessorCount << std::endl;
        std::cout << "Shared memory per mp: " << props.sharedMemPerMultiprocessor << std::endl;
        std::cout << "Max threads per block: " << props.maxThreadsPerBlock << std::endl;
    }
}

namespace SimulatorUtils {
    namespace Structures {
        float &vec3D::operator[](unsigned int ind) { return _vals[ind]; }

        vec3D::vec3D(std::initializer_list<float> list) {
            size_t i = 0;
            for (auto &val: list) {
                _vals[i++] = val;
            }
        }
    }
    namespace Math {
        __host__ __device__ float3 crossProduct(float3 v1, float3 v2) {
            return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
        }

        __host__ __device__ float norm(float3 vec) {

            return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
        }

        __host__ __device__ void assignLinearSpace(float boundary1, float boundary2, size_t steps, float* target, float multiplier = 1){

            auto linearSpan = (boundary2 - boundary1) * multiplier;
            float step = (linearSpan / (float)steps);

            for(size_t i=0; i <= steps; i++){

                target[i] = boundary1 * multiplier + (float)i * step;

            }
        }
    }
}