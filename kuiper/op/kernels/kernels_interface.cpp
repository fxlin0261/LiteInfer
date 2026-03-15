#include "kernels_interface.h"
#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cpu/emb_kernel.h"
#include "cpu/matmul_kernel.h"
#include "cpu/mha_kernel.h"
#include "cpu/rmsnorm_kernel.h"
#include "cpu/rope_kernel.h"
#include "cpu/scale_kernel.h"
#include "cpu/scale_sum_kernel.h"
#include "cpu/softmax_kernel.h"
#include "cpu/swiglu_kernel.h"
#if KUIPER_ENABLE_CUDA
#include "cuda/add_kernel.cuh"
#include "cuda/emb_kernel.cuh"
#include "cuda/matmul_kernel.cuh"
#include "cuda/mha_kernel.cuh"
#include "cuda/rmsnorm_kernel.cuh"
#include "cuda/rope_kernel.cuh"
#include "cuda/swiglu_kernel.cuh"
#endif

namespace kernel {

namespace {
[[noreturn]] void cuda_kernel_unavailable(const char* kernel_name) {
    LOG(FATAL) << kernel_name << " requires CUDA, but this build was compiled without CUDA support.";
}

template <typename Kernel>
Kernel select_kernel(base::DeviceType device_type, Kernel cpu_kernel, Kernel cuda_kernel,
                     const char* cuda_kernel_name, const char* error_message) {
    if (device_type == base::DeviceType::kDeviceCPU && cpu_kernel != nullptr) {
        return cpu_kernel;
    }
    if (device_type == base::DeviceType::kDeviceCUDA) {
        if (cuda_kernel != nullptr) {
            return cuda_kernel;
        }
        cuda_kernel_unavailable(cuda_kernel_name);
    }
    LOG(FATAL) << error_message;
    return nullptr;
}

#if KUIPER_ENABLE_CUDA
constexpr AddKernel kAddKernelCuda = add_kernel_cu;
constexpr EmbeddingKernel kEmbeddingKernelCuda = emb_kernel_cu;
constexpr MatmulKernel kMatmulKernelCuda = matmul_kernel_cu;
constexpr MatmulKernelQuant kMatmulKernelQuant8Cuda = matmul_kernel_cu_qint8;
constexpr MHAKernel kMhaKernelCuda = mha_kernel_cu;
constexpr RoPEKernel kRoPEKernelCuda = rope_kernel_cu;
constexpr SwigluKernel kSwigluKernelCuda = swiglu_kernel_cu;
constexpr RMSNormKernel kRmsNormKernelCuda = rmsnorm_kernel_cu;
constexpr RMSNormKernelDim kRmsNormDimKernelCuda = rmsnorm_kernel_cu_dim;
#else
constexpr AddKernel kAddKernelCuda = nullptr;
constexpr EmbeddingKernel kEmbeddingKernelCuda = nullptr;
constexpr MatmulKernel kMatmulKernelCuda = nullptr;
constexpr MatmulKernelQuant kMatmulKernelQuant8Cuda = nullptr;
constexpr MHAKernel kMhaKernelCuda = nullptr;
constexpr RoPEKernel kRoPEKernelCuda = nullptr;
constexpr SwigluKernel kSwigluKernelCuda = nullptr;
constexpr RMSNormKernel kRmsNormKernelCuda = nullptr;
constexpr RMSNormKernelDim kRmsNormDimKernelCuda = nullptr;
#endif
}  // namespace

AddKernel get_add_kernel(base::DeviceType device_type) {
    return select_kernel(device_type, add_kernel_cpu, kAddKernelCuda, "add_kernel_cu",
                         "Unknown device type for get a add kernel.");
}

EmbeddingKernel get_emb_kernel(base::DeviceType device_type) {
    return select_kernel(device_type, emb_kernel_normal, kEmbeddingKernelCuda, "emb_kernel_cu",
                         "Unknown device type for get an embedding kernel.");
}

MatmulKernel get_matmul_kernel(base::DeviceType device_type) {
    return select_kernel(device_type, matmul_kernel_cpu, kMatmulKernelCuda, "matmul_kernel_cu",
                         "Unknown device type for get an matmul kernel.");
}

MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type) {
    return select_kernel<MatmulKernelQuant>(device_type, nullptr, kMatmulKernelQuant8Cuda,
                                            "matmul_kernel_cu_qint8",
                                            "Unknown device type for get an matmul kernel.");
}

MHAKernel get_mha_kernel(base::DeviceType device_type) {
    return select_kernel(device_type, mha_kernel, kMhaKernelCuda, "mha_kernel_cu",
                         "Unknown device type for get an mha kernel.");
}

RoPEKernel get_rope_kernel(base::DeviceType device_type) {
    return select_kernel(device_type, rope_kernel_cpu, kRoPEKernelCuda, "rope_kernel_cu",
                         "Unknown device type for get a rope kernel.");
}

ScaleKernel get_scale_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_inplace_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a rope kernel.";
        return nullptr;
    }
}

SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return softmax_inplace_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get an softmax kernel.";
        return nullptr;
    }
}

SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void* stream) {
    UNUSED(stream);
    return select_kernel(device_type, swiglu_kernel_cpu, kSwigluKernelCuda, "swiglu_kernel_cu",
                         "Unknown device type for get a swiglu kernel.");
}

RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type) {
    return select_kernel(device_type, rmsnorm_kernel_cpu, kRmsNormKernelCuda,
                         "rmsnorm_kernel_cu", "Unknown device type for get a rmsnorm kernel.");
}

RMSNormKernelDim get_rmsnorm_dim_kernel(base::DeviceType device_type) {
    return select_kernel<RMSNormKernelDim>(device_type, nullptr, kRmsNormDimKernelCuda,
                                           "rmsnorm_kernel_cu_dim",
                                           "Unknown device type for get a rmsnorm dim kernel.");
}

ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return scale_sum_kernel_cpu;
    } else {
        LOG(FATAL) << "Unknown device type for get a scale and reduce kernel.";
        return nullptr;
    }
}

}  // namespace kernel
