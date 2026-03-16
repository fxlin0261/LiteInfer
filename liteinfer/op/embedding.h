#ifndef LITEINFER_INCLUDE_OP_EMBEDDING_H_
#define LITEINFER_INCLUDE_OP_EMBEDDING_H_
#include <utility>
#include "layer.h"

namespace op {
struct EmbeddingOutput {
    tensor::Tensor input_tokens;
    tensor::Tensor input_embeddings;
    int32_t input_token_num = 0;
    explicit EmbeddingOutput(tensor::Tensor input_tokens, tensor::Tensor input_embeddings,
                             int32_t input_token_num)
        : input_tokens(std::move(input_tokens)),
          input_embeddings(std::move(input_embeddings)),
          input_token_num(input_token_num) {}
};

class EmbeddingLayer : public LayerParam {
public:
    explicit EmbeddingLayer(base::DeviceType device_type, int32_t dim, int32_t seq_len,
                            int32_t vocab_size);
    using LayerParam::forward;
    base::Status check() const override;
    base::Status forward() override;

private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};
}  // namespace op
#endif  // LITEINFER_INCLUDE_OP_EMBEDDING_H_
