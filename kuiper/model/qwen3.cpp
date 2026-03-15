#include "model/qwen3.h"
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <utility>

namespace model {

Qwen3Model::Qwen3Model(std::string token_path, std::string model_path, bool is_quant_model)
    : StandardDecoderModel(base::TokenizerType::kEncodeBpe, base::ModelType::kModelTypeQwen3,
                           std::move(token_path), std::move(model_path), is_quant_model) {}

base::Status Qwen3Model::create_param_layers() {
    CHECK(!is_quant_model_);

    size_t pos = 0;
    const int32_t dim = config_->dim_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const int32_t immediate_dim = config_->immediate_dim_;
    const auto cpu_device_type = base::DeviceType::kDeviceCPU;
    float* weight_ptr = reinterpret_cast<float*>(const_cast<void*>(raw_model_data_->weight(pos)));

    for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(
            device_type_, hidden_dim, base::RmsNormEpsilon(model_type_));
        rms_norm_layer->set_weight(0, {hidden_dim}, weight_ptr, cpu_device_type);
        layers().rmsnorm_layers_.push_back(rms_norm_layer);
        weight_ptr += hidden_dim;
    }
    pos += static_cast<size_t>(2 * config_->layer_num_ + 1) * hidden_dim;

    layers().embedding_layer_ =
        std::make_shared<op::EmbeddingLayer>(device_type_, hidden_dim, config_->seq_len_,
                                             config_->vocab_size_);
    layers().embedding_layer_->set_weight(0, {config_->vocab_size_, hidden_dim}, weight_ptr,
                                          cpu_device_type);
    pos += static_cast<size_t>(config_->vocab_size_) * hidden_dim;

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, false);
        wq->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wq_layers_.push_back(wq);
        pos += static_cast<size_t>(hidden_dim) * dim;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(
            device_type_, config_->head_size_, base::RmsNormEpsilon(model_type_));
        rms_norm_layer->set_weight(0, {config_->head_size_}, raw_model_data_->weight(pos),
                                   cpu_device_type);
        layers().query_norm_layers_.push_back(rms_norm_layer);
        pos += config_->head_size_;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, hidden_dim, false);
        wk->set_weight(0, {kv_dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wk_layers_.push_back(wk);
        pos += static_cast<size_t>(hidden_dim) * kv_dim;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(
            device_type_, config_->head_size_, base::RmsNormEpsilon(model_type_));
        rms_norm_layer->set_weight(0, {config_->head_size_}, raw_model_data_->weight(pos),
                                   cpu_device_type);
        layers().key_norm_layers_.push_back(rms_norm_layer);
        pos += config_->head_size_;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, kv_dim, hidden_dim, false);
        wv->set_weight(0, {kv_dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wv_layers_.push_back(wv);
        pos += static_cast<size_t>(hidden_dim) * kv_dim;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, false);
        wo->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wo_layers_.push_back(wo);
        pos += static_cast<size_t>(dim) * hidden_dim;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
        w1->set_weight(0, {immediate_dim, hidden_dim}, raw_model_data_->weight(pos),
                       cpu_device_type);
        layers().w1_layers_.push_back(w1);
        pos += static_cast<size_t>(hidden_dim) * immediate_dim;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, immediate_dim, false);
        w2->set_weight(0, {hidden_dim, immediate_dim}, raw_model_data_->weight(pos),
                       cpu_device_type);
        layers().w2_layers_.push_back(w2);
        pos += static_cast<size_t>(immediate_dim) * hidden_dim;
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, immediate_dim, hidden_dim, false);
        w3->set_weight(0, {immediate_dim, hidden_dim}, raw_model_data_->weight(pos),
                       cpu_device_type);
        layers().w3_layers_.push_back(w3);
        pos += static_cast<size_t>(immediate_dim) * hidden_dim;
    }

    auto lm_head =
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, hidden_dim, false);
    lm_head->set_weight(0, {config_->vocab_size_, hidden_dim}, raw_model_data_->weight(pos),
                        cpu_device_type);
    layers().cls_layer_ = lm_head;
    return base::error::Success();
}

base::Status Qwen3Model::create_param_quant_layers() {
    return base::error::FunctionNotImplement("Qwen3 quantized decoder is not implemented.");
}

bool Qwen3Model::use_qwen_tokenizer() const { return true; }

int32_t Qwen3Model::residual_width() const { return config_->hidden_dim_; }

int32_t Qwen3Model::ffn_width() const { return config_->immediate_dim_; }

base::Status Qwen3Model::validate_custom_layers() const {
    if (layers().query_norm_layers_.size() != config_->layer_num_) {
        return base::error::InternalError(
            "Create the query norm layers for the Qwen3 model failed.");
    }
    if (layers().key_norm_layers_.size() != config_->layer_num_) {
        return base::error::InternalError("Create the key norm layers for the Qwen3 model failed.");
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        if (!layers().query_norm_layers_.at(i)) {
            return base::error::InternalError(
                "Create the query norm layers for the Qwen3 model failed.");
        }
        if (!layers().key_norm_layers_.at(i)) {
            return base::error::InternalError(
                "Create the key norm layers for the Qwen3 model failed.");
        }
    }
    return base::error::Success();
}

void Qwen3Model::apply_attention_projection_norms(int32_t layer_idx, tensor::Tensor& query,
                                                  tensor::Tensor& key) const {
    const auto& query_norm = layers().query_norm_layers_.at(layer_idx);
    const auto& key_norm = layers().key_norm_layers_.at(layer_idx);
    CHECK_NE(query_norm, nullptr) << "The query norm layer in the Qwen3 model is null pointer.";
    CHECK_NE(key_norm, nullptr) << "The key norm layer in the Qwen3 model is null pointer.";

    query.reshape({config_->head_num_, config_->head_size_});
    STATUS_CHECK(query_norm->forward(query, query));
    query.reshape({config_->dim_});

    key.reshape({config_->kv_head_num_, config_->head_size_});
    STATUS_CHECK(key_norm->forward(key, key));
    key.reshape({config_->kv_dim_});
}

}  // namespace model
