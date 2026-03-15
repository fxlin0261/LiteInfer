#ifndef KUIPER_INCLUDE_MODEL_STANDARD_DECODER_H_
#define KUIPER_INCLUDE_MODEL_STANDARD_DECODER_H_

#include <base/cuda_config.h>
#include <memory>
#include <vector>
#include "model/core/model.h"
#include "op/add.h"
#include "op/embedding.h"
#include "op/rope.h"
#include "op/swiglu.h"

namespace model {

struct StandardDecoderLayers {
    std::shared_ptr<op::Layer> add_layer_;
    std::shared_ptr<op::Layer> rope_layer_;
    std::shared_ptr<op::Layer> swiglu_layer_;
    std::shared_ptr<op::Layer> mha_layer_;

    std::vector<std::shared_ptr<op::Layer>> wq_layers_;
    std::vector<std::shared_ptr<op::Layer>> wk_layers_;
    std::vector<std::shared_ptr<op::Layer>> wv_layers_;
    std::vector<std::shared_ptr<op::Layer>> wo_layers_;

    std::vector<std::shared_ptr<op::Layer>> w1_layers_;
    std::vector<std::shared_ptr<op::Layer>> w2_layers_;
    std::vector<std::shared_ptr<op::Layer>> rmsnorm_layers_;
    std::vector<std::shared_ptr<op::Layer>> w3_layers_;
    std::vector<std::shared_ptr<op::Layer>> query_norm_layers_;
    std::vector<std::shared_ptr<op::Layer>> key_norm_layers_;
    std::shared_ptr<op::Layer> cls_layer_;

    std::shared_ptr<op::Layer> embedding_layer_;

    void to_cuda(std::shared_ptr<kernel::CudaConfig> config);
};

class StandardDecoderModel : public Model {
public:
    explicit StandardDecoderModel(base::TokenizerType tokenizer_type, base::ModelType model_type,
                                  std::string token_path, std::string model_path,
                                  bool is_quant_model);

    base::Status init(base::DeviceType device_type) override;

    base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         bool is_prompt, int& next) const override;

    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         int& next) const override;

    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override;

protected:
    StandardDecoderLayers& layers();

    const StandardDecoderLayers& layers() const;

    const std::shared_ptr<kernel::CudaConfig>& cuda_config() const;

    base::Status create_param_layers() override = 0;

    int32_t input_width() const override;

    virtual int32_t residual_width() const;

    virtual int32_t attention_width() const;

    virtual int32_t ffn_width() const;

    virtual base::Status validate_custom_layers() const;

    virtual void apply_attention_projection_norms(int32_t layer_idx, tensor::Tensor& query,
                                                  tensor::Tensor& key) const;

private:
    void init_mem() override;

    base::Status create_layers() override;

    base::Status create_nonparam_layers() override;

    base::Status create_param_quant_layers() override;

    void attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

    void attention_rms(int32_t layer_idx, const tensor::Tensor& input) const;

    void feed_forward(int32_t layer_idx, const tensor::Tensor& input) const;

    void attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const;

    void cls_logits(const tensor::Tensor& input) const;

    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override;

private:
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
    std::unique_ptr<StandardDecoderLayers> layers_;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_STANDARD_DECODER_H_
