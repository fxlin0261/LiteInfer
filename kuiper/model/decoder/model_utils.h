#ifndef KUIPER_INCLUDE_MODEL_MODEL_UTILS_H_
#define KUIPER_INCLUDE_MODEL_MODEL_UTILS_H_

#include <memory>
#include <vector>
#include "base/cuda_config.h"
#include "op/layer.h"
#include "tensor/tensor.h"

namespace model {
namespace detail {

base::Status InitCudaConfig(std::shared_ptr<kernel::CudaConfig>& cuda_config);

base::Status InitSinCosCache(base::ModelType model_type, int32_t head_size, int32_t seq_len,
                             const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                             const std::shared_ptr<kernel::CudaConfig>& cuda_config);

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

#endif  // KUIPER_INCLUDE_MODEL_MODEL_UTILS_H_
