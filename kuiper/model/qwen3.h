#ifndef KUIPER_INCLUDE_MODEL_QWEN3_H_
#define KUIPER_INCLUDE_MODEL_QWEN3_H_

#include "standard_decoder.h"

namespace model {

class Qwen3Model : public StandardDecoderModel {
public:
    explicit Qwen3Model(std::string token_path, std::string model_path, bool is_quant_model);

private:
    base::Status create_param_layers() override;

    base::Status create_param_quant_layers() override;

    bool use_qwen_tokenizer() const override;

    int32_t residual_width() const override;

    int32_t ffn_width() const override;

    base::Status validate_custom_layers() const override;

    void apply_attention_projection_norms(int32_t layer_idx, tensor::Tensor& query,
                                          tensor::Tensor& key) const override;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN3_H_
