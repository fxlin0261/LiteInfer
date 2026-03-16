#include "model/llama/llama.h"

#include <cstdlib>
#include <utility>

#include <op/matmul.h>
#include <op/rmsnorm.h>

#include "model/decoder/model_utils.h"

namespace model {
namespace {
struct DenseWeightsLayout {
    size_t embedding = 0;
    size_t attn_rms = 0;
    size_t wq = 0;
    size_t wk = 0;
    size_t wv = 0;
    size_t wo = 0;
    size_t ffn_rms = 0;
    size_t w1 = 0;
    size_t w2 = 0;
    size_t w3 = 0;
    size_t final_rms = 0;
    size_t rope_cache = 0;
    size_t cls = 0;
};

void ReserveLayerStorage(StandardDecoderLayers& layers, int32_t layer_num) {
    layers.wq_layers_.reserve(layer_num);
    layers.wk_layers_.reserve(layer_num);
    layers.wv_layers_.reserve(layer_num);
    layers.wo_layers_.reserve(layer_num);
    layers.w1_layers_.reserve(layer_num);
    layers.w2_layers_.reserve(layer_num);
    layers.w3_layers_.reserve(layer_num);
    layers.rmsnorm_layers_.reserve(2 * layer_num + 1);
}

std::shared_ptr<op::MatmulLayer> CreateMatmulLayer(base::DeviceType device_type,
                                                   int32_t out_features, int32_t in_features,
                                                   const void* weight_ptr,
                                                   base::DeviceType weight_device) {
    auto layer = std::make_shared<op::MatmulLayer>(device_type, out_features, in_features);
    const auto status = layer->set_weight(0, {out_features, in_features}, weight_ptr, weight_device);
    CHECK(status.ok()) << status.message();
    return layer;
}

std::shared_ptr<op::MatmulLayer> CreateQuantizedMatmulLayer(base::DeviceType device_type,
                                                            int32_t out_features,
                                                            int32_t in_features,
                                                            const void* weight_ptr,
                                                            base::DeviceType weight_device,
                                                            int32_t group_size) {
    auto layer =
        std::make_shared<op::MatmulLayer>(device_type, out_features, in_features, true);
    layer->set_group_size(group_size);
    const auto status = layer->set_weight(0, {out_features, in_features}, weight_ptr, weight_device);
    CHECK(status.ok()) << status.message();
    return layer;
}

std::shared_ptr<op::EmbeddingLayer> CreateEmbeddingLayer(base::DeviceType device_type, int32_t dim,
                                                         int32_t seq_len, int32_t vocab_size,
                                                         const void* weight_ptr,
                                                         base::DeviceType weight_device) {
    auto layer = std::make_shared<op::EmbeddingLayer>(device_type, dim, seq_len, vocab_size);
    const auto status = layer->set_weight(0, {vocab_size, dim}, weight_ptr, weight_device);
    CHECK(status.ok()) << status.message();
    return layer;
}

std::shared_ptr<op::RmsNormLayer> CreateRmsNormLayer(base::DeviceType device_type,
                                                     base::ModelType model_type, int32_t dim,
                                                     const void* weight_ptr,
                                                     base::DeviceType weight_device) {
    auto layer =
        std::make_shared<op::RmsNormLayer>(device_type, dim, base::RmsNormEpsilon(model_type));
    const auto status = layer->set_weight(0, {dim}, weight_ptr, weight_device);
    CHECK(status.ok()) << status.message();
    return layer;
}

size_t AppendQuantizedMatmulLayers(const RawModelData& raw_model_data, int32_t layer_num,
                                   int32_t out_features, int32_t in_features, size_t offset,
                                   int32_t group_size, base::DeviceType device_type,
                                   base::DeviceType weight_device,
                                   std::vector<std::shared_ptr<op::Layer>>& layers) {
    for (int32_t layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
        layers.push_back(CreateQuantizedMatmulLayer(
            device_type, out_features, in_features, raw_model_data.weight(offset), weight_device,
            group_size));
        offset += detail::LegacyQuantizedTensorBytes(out_features, in_features, group_size);
    }
    return offset;
}

DenseWeightsLayout BuildDenseWeightsLayout(const TransformerConfig& config) {
    const int32_t dim = config.dim_;
    const int32_t hidden_dim = config.hidden_dim_;
    const int32_t layer_num = config.layer_num_;
    const int32_t kv_dim = config.kv_dim_;
    const int32_t vocab_size = config.vocab_size_;
    const int32_t seq_len = config.seq_len_;
    const int32_t head_size = config.head_size_;

    DenseWeightsLayout layout;
    size_t offset = 0;
    layout.embedding = offset;
    offset += static_cast<size_t>(vocab_size) * dim;
    layout.attn_rms = offset;
    offset += static_cast<size_t>(layer_num) * dim;
    layout.wq = offset;
    offset += static_cast<size_t>(layer_num) * dim * dim;
    layout.wk = offset;
    offset += static_cast<size_t>(layer_num) * kv_dim * dim;
    layout.wv = offset;
    offset += static_cast<size_t>(layer_num) * kv_dim * dim;
    layout.wo = offset;
    offset += static_cast<size_t>(layer_num) * dim * dim;
    layout.ffn_rms = offset;
    offset += static_cast<size_t>(layer_num) * dim;
    layout.w1 = offset;
    offset += static_cast<size_t>(layer_num) * hidden_dim * dim;
    layout.w2 = offset;
    offset += static_cast<size_t>(layer_num) * hidden_dim * dim;
    layout.w3 = offset;
    offset += static_cast<size_t>(layer_num) * hidden_dim * dim;
    layout.final_rms = offset;
    offset += dim;
    layout.rope_cache = offset;
    offset += static_cast<size_t>(seq_len) * head_size;
    layout.cls = offset;
    return layout;
}
}  // namespace

LlamaModelBase::LlamaModelBase(base::TokenizerType tokenizer_type, base::ModelType model_type,
                               std::string token_path, std::string model_path, bool is_quant_model)
    : StandardDecoderModel(tokenizer_type, model_type, std::move(token_path),
                           std::move(model_path), is_quant_model) {}

base::Status LlamaModelBase::create_param_quant_layers() {
    CHECK(is_quant_model_);
    auto& decoder_layers = layers();
    ReserveLayerStorage(decoder_layers, config_->layer_num_);

    const auto weight_device = base::DeviceType::kDeviceCPU;
    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const int32_t layer_num = config_->layer_num_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t vocab_size = std::abs(config_->vocab_size_);

    size_t offset = 0;
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.wq_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, kv_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.wk_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, kv_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.wv_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.wo_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, hidden_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.w1_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, dim, hidden_dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.w2_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, hidden_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         decoder_layers.w3_layers_);

    const auto cls_and_embedding = detail::ResolveLegacyQuantizedWeightsLayout(
        *raw_model_data_, offset, vocab_size, dim, group_size_, config_->is_shared_weight_);
    if (cls_and_embedding.classifier_is_quantized) {
        decoder_layers.cls_layer_ = CreateQuantizedMatmulLayer(
            device_type_, vocab_size, dim, cls_and_embedding.classifier_weight, weight_device,
            group_size_);
    } else {
        decoder_layers.cls_layer_ = CreateMatmulLayer(
            device_type_, vocab_size, dim, cls_and_embedding.classifier_weight, weight_device);
    }

    auto* weight_cursor =
        reinterpret_cast<float*>(const_cast<void*>(cls_and_embedding.embedding_weight));
    decoder_layers.embedding_layer_ =
        CreateEmbeddingLayer(device_type_, dim, config_->seq_len_, vocab_size, weight_cursor,
                             weight_device);
    weight_cursor += static_cast<size_t>(vocab_size) * dim;

    const int32_t rms_layer_count = 2 * layer_num + 1;
    for (int32_t layer_idx = 0; layer_idx < rms_layer_count; ++layer_idx) {
        decoder_layers.rmsnorm_layers_.push_back(
            CreateRmsNormLayer(device_type_, model_type_, dim, weight_cursor, weight_device));
        weight_cursor += dim;
    }
    return base::error::Success();
}

base::Status LlamaModelBase::create_param_layers() {
    CHECK(!is_quant_model_);
    auto& decoder_layers = layers();
    ReserveLayerStorage(decoder_layers, config_->layer_num_);

    const auto weight_device = base::DeviceType::kDeviceCPU;
    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const int32_t layer_num = config_->layer_num_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t vocab_size = std::abs(config_->vocab_size_);
    const DenseWeightsLayout layout = BuildDenseWeightsLayout(*config_);

    decoder_layers.embedding_layer_ = CreateEmbeddingLayer(
        device_type_, dim, config_->seq_len_, vocab_size, raw_model_data_->weight(layout.embedding),
        weight_device);

    for (int32_t layer_idx = 0; layer_idx < layer_num; ++layer_idx) {
        const size_t attn_rms_offset = layout.attn_rms + static_cast<size_t>(layer_idx) * dim;
        const size_t wq_offset = layout.wq + static_cast<size_t>(layer_idx) * dim * dim;
        const size_t wk_offset = layout.wk + static_cast<size_t>(layer_idx) * kv_dim * dim;
        const size_t wv_offset = layout.wv + static_cast<size_t>(layer_idx) * kv_dim * dim;
        const size_t wo_offset = layout.wo + static_cast<size_t>(layer_idx) * dim * dim;
        const size_t ffn_rms_offset = layout.ffn_rms + static_cast<size_t>(layer_idx) * dim;
        const size_t w1_offset = layout.w1 + static_cast<size_t>(layer_idx) * hidden_dim * dim;
        const size_t w2_offset = layout.w2 + static_cast<size_t>(layer_idx) * hidden_dim * dim;
        const size_t w3_offset = layout.w3 + static_cast<size_t>(layer_idx) * hidden_dim * dim;

        decoder_layers.rmsnorm_layers_.push_back(CreateRmsNormLayer(
            device_type_, model_type_, dim, raw_model_data_->weight(attn_rms_offset),
            weight_device));
        decoder_layers.wq_layers_.push_back(
            CreateMatmulLayer(device_type_, dim, dim, raw_model_data_->weight(wq_offset),
                              weight_device));
        decoder_layers.wk_layers_.push_back(
            CreateMatmulLayer(device_type_, kv_dim, dim, raw_model_data_->weight(wk_offset),
                              weight_device));
        decoder_layers.wv_layers_.push_back(
            CreateMatmulLayer(device_type_, kv_dim, dim, raw_model_data_->weight(wv_offset),
                              weight_device));
        decoder_layers.wo_layers_.push_back(
            CreateMatmulLayer(device_type_, dim, dim, raw_model_data_->weight(wo_offset),
                              weight_device));
        decoder_layers.rmsnorm_layers_.push_back(CreateRmsNormLayer(
            device_type_, model_type_, dim, raw_model_data_->weight(ffn_rms_offset),
            weight_device));
        decoder_layers.w1_layers_.push_back(
            CreateMatmulLayer(device_type_, hidden_dim, dim, raw_model_data_->weight(w1_offset),
                              weight_device));
        decoder_layers.w2_layers_.push_back(CreateMatmulLayer(
            device_type_, dim, hidden_dim, raw_model_data_->weight(w2_offset), weight_device));
        decoder_layers.w3_layers_.push_back(
            CreateMatmulLayer(device_type_, hidden_dim, dim, raw_model_data_->weight(w3_offset),
                              weight_device));
    }

    decoder_layers.rmsnorm_layers_.push_back(CreateRmsNormLayer(
        device_type_, model_type_, dim, raw_model_data_->weight(layout.final_rms), weight_device));

    const void* cls_weight = config_->is_shared_weight_ ? raw_model_data_->weight(layout.embedding)
                                                        : raw_model_data_->weight(layout.cls);
    decoder_layers.cls_layer_ =
        CreateMatmulLayer(device_type_, vocab_size, dim, cls_weight, weight_device);
    return base::error::Success();
}
}  // namespace model
