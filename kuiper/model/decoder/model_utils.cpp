#include "model/decoder/model_utils.h"

#if KUIPER_ENABLE_CUDA
#include "op/kernels/cuda/rope_kernel.cuh"
#endif

namespace model {
namespace detail {
size_t LegacyQuantizedTensorBytes(int32_t rows, int32_t cols, int32_t group_size) {
    CHECK_GT(rows, 0);
    CHECK_GT(cols, 0);
    CHECK_GT(group_size, 0);
    const size_t weight_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    CHECK_EQ(weight_elements % static_cast<size_t>(group_size), 0U);
    const size_t scale_count = weight_elements / static_cast<size_t>(group_size);
    return weight_elements + scale_count * sizeof(float);
}

LegacyQuantizedWeightsLayout ResolveLegacyQuantizedWeightsLayout(
    const RawModelData& raw_model_data, size_t offset, int32_t vocab_size, int32_t dim,
    int32_t group_size, bool shared_classifier) {
    CHECK_GT(vocab_size, 0);
    CHECK_GT(dim, 0);
    CHECK_GT(group_size, 0);

    LegacyQuantizedWeightsLayout layout;
    layout.classifier_weight = raw_model_data.weight(offset);
    if (shared_classifier) {
        layout.embedding_weight = layout.classifier_weight;
        layout.classifier_is_quantized = false;
        return layout;
    }

    layout.classifier_is_quantized = true;
    layout.embedding_weight =
        raw_model_data.weight(offset + LegacyQuantizedTensorBytes(vocab_size, dim, group_size));
    return layout;
}

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
