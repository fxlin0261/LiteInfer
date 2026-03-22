#ifndef LITEINFER_INCLUDE_MODEL_LLAMA_MODEL_UTILS_H_
#define LITEINFER_INCLUDE_MODEL_LLAMA_MODEL_UTILS_H_

#include <memory>
#include <vector>
#include "base/cuda_config.h"
#include "model/raw_model_data.h"
#include "op/layer.h"
#include "base/tensor.h"

namespace model {
namespace detail {
struct LegacyQuantizedWeightsLayout {
    const void* classifier_weight = nullptr;
    const void* embedding_weight = nullptr;
    bool classifier_is_quantized = false;
};

base::Status InitCudaConfig(std::shared_ptr<kernel::CudaConfig>& cuda_config);
base::Status InitSinCosCache(float rope_theta, const base::RoPEScalingConfig& rope_scaling,
                             int32_t head_size, int32_t seq_len,
                             const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                             const std::shared_ptr<kernel::CudaConfig>& cuda_config);
size_t LegacyQuantizedTensorBytes(int32_t rows, int32_t cols, int32_t group_size);
LegacyQuantizedWeightsLayout ResolveLegacyQuantizedWeightsLayout(
    const RawModelData& raw_model_data, size_t offset, int32_t vocab_size, int32_t dim,
    int32_t group_size, bool shared_classifier);

inline void MoveLayerToCuda(const std::shared_ptr<op::Layer>& layer,
                            const std::shared_ptr<kernel::CudaConfig>& config) {
    if (!layer) {
        return;
    }
    layer->set_cuda_config(config);
    layer->to_cuda();
}

template <typename LayerCollection>
void MoveLayerRangeToCuda(const LayerCollection& layers,
                          const std::shared_ptr<kernel::CudaConfig>& config) {
    for (const auto& layer : layers) {
        MoveLayerToCuda(layer, config);
    }
}
}  // namespace detail
}  // namespace model

#endif  // LITEINFER_INCLUDE_MODEL_LLAMA_MODEL_UTILS_H_
