#include "model/model_utils.h"

#if KUIPER_ENABLE_CUDA
#include "op/kernels/cuda/rope_kernel.cuh"
#endif

namespace model {
namespace detail {

base::Status InitCudaConfig(std::shared_ptr<kernel::CudaConfig>& cuda_config) {
#if KUIPER_ENABLE_CUDA
    cudaSetDevice(0);
    cuda_config = std::make_shared<kernel::CudaConfig>();
    cudaStreamCreate(&cuda_config->stream);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return base::error::InternalError("The cuda handle create failed.");
    }
    return base::error::Success();
#else
    UNUSED(cuda_config);
    return base::error::InternalError("This build does not include CUDA support.");
#endif
}

base::Status InitSinCosCache(base::ModelType model_type, int32_t head_size, int32_t seq_len,
                             const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                             const std::shared_ptr<kernel::CudaConfig>& cuda_config) {
#if KUIPER_ENABLE_CUDA
    CHECK_NE(cuda_config, nullptr);
    kernel::sin_cos_cache_calc_cu(model_type, head_size, seq_len, sin_cache, cos_cache,
                                  cuda_config->stream);
    return base::error::Success();
#else
    UNUSED(model_type);
    UNUSED(head_size);
    UNUSED(seq_len);
    UNUSED(sin_cache);
    UNUSED(cos_cache);
    UNUSED(cuda_config);
    return base::error::InternalError("This build does not include CUDA support.");
#endif
}

}  // namespace detail
}  // namespace model
