#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H

#include "base.h"

#ifndef LITEINFER_ENABLE_CUDA
#define LITEINFER_ENABLE_CUDA 0
#endif

namespace base {
inline constexpr bool IsCudaEnabled() { return LITEINFER_ENABLE_CUDA != 0; }

inline constexpr DeviceType DefaultDeviceType() {
    return IsCudaEnabled() ? DeviceType::kDeviceCUDA : DeviceType::kDeviceCPU;
}
}  // namespace base

#if LITEINFER_ENABLE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace kernel {
struct CudaConfig {
    cudaStream_t stream = nullptr;
    ~CudaConfig() {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
};
}  // namespace kernel
#else
using cudaStream_t = void*;
namespace kernel {
struct CudaConfig {
    void* stream = nullptr;
};
}  // namespace kernel
#endif

#endif  // BLAS_HELPER_H
