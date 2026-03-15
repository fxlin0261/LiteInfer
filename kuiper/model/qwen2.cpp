#include "model/qwen2.h"
#include <op/matmul.h>
#include <op/rmsnorm.h>
#include <utility>

namespace model {

Qwen2Model::Qwen2Model(std::string token_path, std::string model_path, bool is_quant_model)
    : StandardDecoderModel(base::TokenizerType::kEncodeBpe, base::ModelType::kModelTypeQwen2,
                           std::move(token_path), std::move(model_path), is_quant_model) {}

base::Status Qwen2Model::create_param_layers() {
    CHECK(!is_quant_model_);

    auto cpu_device_type = base::DeviceType::kDeviceCPU;
    layers().embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, config_->vocab_size_);
    layers().embedding_layer_->set_weight(0, {config_->vocab_size_, config_->dim_},
                                          raw_model_data_->weight(0), cpu_device_type);

    int32_t dim = config_->dim_;
    size_t pos = static_cast<size_t>(dim) * config_->vocab_size_ +
                 static_cast<size_t>(dim) * config_->layer_num_;

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, false, true);
        wq->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(dim) * dim;
        wq->set_bias(0, dim, raw_model_data_->weight(pos), cpu_device_type);
        pos += dim;
        layers().wq_layers_.push_back(wq);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk =
            std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
        wk->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(config_->kv_dim_) * dim;
        int32_t kv_dim = config_->kv_dim_;
        wk->set_bias(0, kv_dim, raw_model_data_->weight(pos), cpu_device_type);
        pos += config_->kv_dim_;
        layers().wk_layers_.push_back(wk);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv =
            std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, false, true);
        wv->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(config_->kv_dim_) * dim;
        int32_t kv_dim = config_->kv_dim_;
        wv->set_bias(0, kv_dim, raw_model_data_->weight(pos), cpu_device_type);
        pos += config_->kv_dim_;
        layers().wv_layers_.push_back(wv);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim);
        wo->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(dim) * dim;
        layers().wo_layers_.push_back(wo);
    }

    pos += static_cast<size_t>(config_->layer_num_) * dim;

    int32_t hidden_dim = config_->hidden_dim_;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w1->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(dim) * hidden_dim;
        layers().w1_layers_.push_back(w1);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim);
        w2->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(dim) * hidden_dim;
        layers().w2_layers_.push_back(w2);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim);
        w3->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        pos += static_cast<size_t>(dim) * hidden_dim;
        layers().w3_layers_.push_back(w3);
    }

    pos += dim;
    pos += static_cast<size_t>(config_->seq_len_) * config_->head_size_;

    layers().cls_layer_ =
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
    if (config_->is_shared_weight_) {
        layers().cls_layer_->set_weight(0, {config_->vocab_size_, dim}, raw_model_data_->weight(0),
                                        cpu_device_type);
    } else {
        layers().cls_layer_->set_weight(0, {config_->vocab_size_, dim}, raw_model_data_->weight(pos),
                                        cpu_device_type);
    }

    size_t rmsnorm_pos = static_cast<size_t>(config_->dim_) * config_->vocab_size_;
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(
            device_type_, config_->dim_, base::RmsNormEpsilon(model_type_));
        rms_norm_layer->set_weight(0, {config_->dim_}, raw_model_data_->weight(rmsnorm_pos),
                                   cpu_device_type);
        layers().rmsnorm_layers_.push_back(rms_norm_layer);
        rmsnorm_pos += config_->dim_;
    }

    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) *
                   (static_cast<size_t>(config_->dim_) * config_->dim_ + config_->dim_);
    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) *
                   (static_cast<size_t>(config_->dim_) * config_->kv_dim_ + config_->kv_dim_);
    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) *
                   (static_cast<size_t>(config_->dim_) * config_->kv_dim_ + config_->kv_dim_);
    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) * config_->dim_ * config_->dim_;

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(
            device_type_, config_->dim_, base::RmsNormEpsilon(model_type_));
        rms_norm_layer->set_weight(0, {config_->dim_}, raw_model_data_->weight(rmsnorm_pos),
                                   cpu_device_type);
        layers().rmsnorm_layers_.push_back(rms_norm_layer);
        rmsnorm_pos += config_->dim_;
    }

    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) * config_->hidden_dim_ * config_->dim_;
    rmsnorm_pos += static_cast<size_t>(config_->layer_num_) * config_->hidden_dim_ * config_->dim_;

    auto rms_final_layer = std::make_shared<op::RmsNormLayer>(
        device_type_, config_->dim_, base::RmsNormEpsilon(model_type_));
    rms_final_layer->set_weight(0, {config_->dim_}, raw_model_data_->weight(rmsnorm_pos),
                                cpu_device_type);
    layers().rmsnorm_layers_.push_back(rms_final_layer);
    return base::error::Success();
}

bool Qwen2Model::use_qwen_tokenizer() const { return true; }

}  // namespace model
