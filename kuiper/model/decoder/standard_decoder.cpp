#include "model/decoder/standard_decoder.h"
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include "model/decoder/model_utils.h"
#include "op/kernels/cpu/rope_kernel.h"

namespace model {

void StandardDecoderLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    detail::MoveLayerToCuda(add_layer_, config);
    detail::MoveLayerToCuda(rope_layer_, config);
    detail::MoveLayerToCuda(swiglu_layer_, config);
    detail::MoveLayerToCuda(cls_layer_, config);
    detail::MoveLayerToCuda(embedding_layer_, config);
    detail::MoveLayerToCuda(mha_layer_, config);
    detail::MoveLayerRangeToCuda(wq_layers_, config);
    detail::MoveLayerRangeToCuda(wk_layers_, config);
    detail::MoveLayerRangeToCuda(wv_layers_, config);
    detail::MoveLayerRangeToCuda(wo_layers_, config);
    detail::MoveLayerRangeToCuda(w1_layers_, config);
    detail::MoveLayerRangeToCuda(w2_layers_, config);
    detail::MoveLayerRangeToCuda(w3_layers_, config);
    detail::MoveLayerRangeToCuda(rmsnorm_layers_, config);
    detail::MoveLayerRangeToCuda(query_norm_layers_, config);
    detail::MoveLayerRangeToCuda(key_norm_layers_, config);
}

StandardDecoderModel::StandardDecoderModel(base::TokenizerType tokenizer_type,
                                           base::ModelType model_type, std::string token_path,
                                           std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, model_type, std::move(token_path), std::move(model_path),
            is_quant_model) {}

base::Status StandardDecoderModel::init(base::DeviceType device_type) {
    using namespace base;
    if (token_path_.empty()) {
        return error::PathNotValid(token_path_);
    }
    if (device_type == DeviceType::kDeviceCPU && is_quant_model_) {
        return error::InternalError("The cpu device do not support int8 quant model.");
    }

    device_type_ = device_type;
    if (device_type == DeviceType::kDeviceCUDA) {
        auto cuda_status = detail::InitCudaConfig(cuda_config_);
        if (!cuda_status) {
            return cuda_status;
        }
    }

    auto read_status = gen_model_from_file();
    if (!read_status) {
        return read_status;
    }

    init_mem();
    if (device_type_ == DeviceType::kDeviceCPU) {
        kernel::sin_cos_cache_calc_cpu(model_type_, config_->head_size_, config_->seq_len_,
                                       get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                       get_buffer(ModelBufferType::kCosCache).ptr<float>());
    } else {
        auto cache_status =
            detail::InitSinCosCache(model_type_, config_->head_size_, config_->seq_len_,
                                    get_buffer(ModelBufferType::kSinCache),
                                    get_buffer(ModelBufferType::kCosCache), cuda_config_);
        if (!cache_status) {
            return cache_status;
        }
    }

    sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
    return error::Success();
}

base::Status StandardDecoderModel::predict(const tensor::Tensor& input,
                                           const tensor::Tensor& pos_tensor, bool is_prompt,
                                           int& next) const {
    auto status = forward(input, pos_tensor, next);
    if (!status) {
        return status;
    }
    next = post_processing(pos_tensor, is_prompt);
    return base::error::Success();
}

base::Status StandardDecoderModel::forward(const tensor::Tensor& input,
                                           const tensor::Tensor& pos_tensor, int& next) const {
    // Run the shared decoder stack over one prompt/generation step.
    UNUSED(next);
    if (input.is_empty()) {
        return base::error::InvalidArgument("The input tensor is empty.");
    }
    if (device_type_ == base::DeviceType::kDeviceCPU && is_quant_model_) {
        return base::error::InternalError("Unsupported int8 quant in the cpu device");
    }

    for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx) {
        attention_rms(layer_idx, input);
        attention_qkv(layer_idx, pos_tensor);
        attention_mha(layer_idx, pos_tensor);
        feed_forward(layer_idx, input);
    }
    cls_logits(input);
    return base::error::Success();
}

op::EmbeddingOutput StandardDecoderModel::embedding(const std::vector<int>& tokens) const {
    // 将 token id 转成 decoder 输入 embedding。
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
    auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
    const int32_t embedding_width = residual_width();
    if (input_tokens.size() != tokens.size()) {
        input_tokens.reshape({static_cast<int32_t>(tokens.size())});
        input_embeddings.reshape({static_cast<int32_t>(tokens.size()), embedding_width});
    }
    for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
        input_tokens.index<int32_t>(i) = tokens.at(i);
    }

    auto input_token_num =
        tensor::Tensor(base::DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
    LOG_IF(FATAL, !layers().embedding_layer_)
        << "The embedding layer in the decoder model is null pointer.";
    STATUS_CHECK(
        layers().embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

    return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
}

StandardDecoderLayers& StandardDecoderModel::layers() {
    CHECK(layers_ != nullptr) << "The decoder layers are null pointer.";
    return *layers_;
}

const StandardDecoderLayers& StandardDecoderModel::layers() const {
    CHECK(layers_ != nullptr) << "The decoder layers are null pointer.";
    return *layers_;
}

const std::shared_ptr<kernel::CudaConfig>& StandardDecoderModel::cuda_config() const {
    return cuda_config_;
}

int32_t StandardDecoderModel::input_width() const { return residual_width(); }

int32_t StandardDecoderModel::residual_width() const { return config_->dim_; }

int32_t StandardDecoderModel::attention_width() const { return config_->dim_; }

int32_t StandardDecoderModel::ffn_width() const { return config_->hidden_dim_; }

base::Status StandardDecoderModel::validate_custom_layers() const { return base::error::Success(); }

void StandardDecoderModel::apply_attention_projection_norms(int32_t layer_idx,
                                                            tensor::Tensor& query,
                                                            tensor::Tensor& key) const {
    UNUSED(layer_idx);
    UNUSED(query);
    UNUSED(key);
}

void StandardDecoderModel::init_mem() {
    // 分配 embedding、中间激活和 KV cache 等运行期缓冲区。
    std::shared_ptr<base::DeviceAllocator> alloc;
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        alloc = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
        alloc = base::CUDADeviceAllocatorFactory::get_instance();
    }

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
        layers().to_cuda(cuda_config_);
    }

    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();
    const int32_t residual_dim = residual_width();
    const int32_t attention_dim = attention_width();
    const int32_t ffn_dim = ffn_width();

    tensor::Tensor input_tokens(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32, 1, residual_dim, true, alloc);
    tensor::Tensor sin_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                             true, alloc);
    tensor::Tensor cos_cache(base::DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                             true, alloc);

    CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
    CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

    tensor::Tensor rms_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    tensor::Tensor mha_output(base::DataType::kDataTypeFp32, attention_dim, true, alloc);
    tensor::Tensor w2_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    tensor::Tensor ffn_rms_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
    CHECK(insert_buffer(ModelBufferType::kOutputMHA, mha_output));
    CHECK(insert_buffer(ModelBufferType::kW2Output, w2_output));
    CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, ffn_rms_output));

    tensor::Tensor w1_output(base::DataType::kDataTypeFp32, ffn_dim, true, alloc);
    tensor::Tensor w3_output(base::DataType::kDataTypeFp32, ffn_dim, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
    CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

    tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
    tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_,
                               config_->seq_len_, config_->kv_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

    tensor::Tensor query(base::DataType::kDataTypeFp32, attention_dim, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kQuery, query));

    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

    tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                        alloc);
    CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
    tensor::Tensor attn_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kAttnOutput, attn_output));

    tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                          alloc_cpu);
        CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
    }
    CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

base::Status StandardDecoderModel::create_layers() {
    using namespace base;
    if (!layers_) {
        layers_ = std::make_unique<StandardDecoderLayers>();
    }

    base::Status layer_status =
        !is_quant_model_ ? create_param_layers() : create_param_quant_layers();
    if (!layer_status) {
        return layer_status;
    }

    layer_status = create_nonparam_layers();
    if (!layer_status) {
        return layer_status;
    }

    if (!layers().embedding_layer_) {
        return error::InternalError("Create the embedding layer for the decoder model failed!");
    }

    if (layers().rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1) {
        return error::InternalError("Create the rmsnorm layers for the decoder model failed!");
    }

    if (layers().wq_layers_.size() != config_->layer_num_ ||
        layers().wk_layers_.size() != config_->layer_num_ ||
        layers().wv_layers_.size() != config_->layer_num_ ||
        layers().wo_layers_.size() != config_->layer_num_) {
        return error::InternalError(
            "Create the attention matmul layers for the decoder model "
            "failed.");
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        if (!layers().wq_layers_.at(i) || !layers().wk_layers_.at(i) ||
            !layers().wv_layers_.at(i) || !layers().wo_layers_.at(i)) {
            return error::InternalError(
                "Create the attention layers for the decoder model "
                "failed.");
        }
    }

    if (layers().w1_layers_.size() != config_->layer_num_ ||
        layers().w2_layers_.size() != config_->layer_num_ ||
        layers().w3_layers_.size() != config_->layer_num_) {
        return error::InternalError(
            "Create the feedforward matmul layers for the decoder model "
            "failed.");
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        if (!layers().w1_layers_.at(i) || !layers().w2_layers_.at(i) ||
            !layers().w3_layers_.at(i)) {
            return error::InternalError(
                "Create the feedforward layers for the decoder model "
                "failed.");
        }
    }

    if (!layers().rope_layer_ || !layers().add_layer_ || !layers().mha_layer_ ||
        !layers().swiglu_layer_) {
        return error::InternalError(
            "Create the non-parameter layers for the decoder model "
            "failed.");
    }

    if (!layers().query_norm_layers_.empty() &&
        layers().query_norm_layers_.size() != config_->layer_num_) {
        return error::InternalError("Create the query norm layers for the decoder model failed.");
    }

    if (!layers().key_norm_layers_.empty() &&
        layers().key_norm_layers_.size() != config_->layer_num_) {
        return error::InternalError("Create the key norm layers for the decoder model failed.");
    }

    for (int32_t i = 0; i < static_cast<int32_t>(layers().query_norm_layers_.size()); ++i) {
        if (!layers().query_norm_layers_.at(i)) {
            return error::InternalError(
                "Create the query norm layers for the decoder model failed.");
        }
    }

    for (int32_t i = 0; i < static_cast<int32_t>(layers().key_norm_layers_.size()); ++i) {
        if (!layers().key_norm_layers_.at(i)) {
            return error::InternalError("Create the key norm layers for the decoder model failed.");
        }
    }

    return validate_custom_layers();
}

base::Status StandardDecoderModel::create_nonparam_layers() {
    layers().rope_layer_ = std::make_shared<op::RoPELayer>(device_type_, model_type_, config_->dim_,
                                                           config_->kv_dim_, config_->head_size_);

    layers().mha_layer_ = std::make_shared<op::MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
        config_->head_size_);

    layers().add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
    layers().swiglu_layer_ = std::make_shared<op::SwiGLULayer>(device_type_, ffn_width());
    return base::error::Success();
}

base::Status StandardDecoderModel::create_param_quant_layers() {
    CHECK(is_quant_model_);

    size_t pos = 0;
    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const auto cpu_device_type = base::DeviceType::kDeviceCPU;

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
        wq->set_group_size(group_size_);
        wq->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wq_layers_.push_back(wq);
        pos += dim * dim + wq->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
        wk->set_group_size(group_size_);
        wk->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wk_layers_.push_back(wk);
        pos += config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
        wv->set_group_size(group_size_);
        wv->set_weight(0, {config_->kv_dim_, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
        wo->set_group_size(group_size_);
        wo->set_weight(0, {dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().wo_layers_.push_back(wo);
        pos += dim * dim + wo->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w1->set_group_size(group_size_);
        w1->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().w1_layers_.push_back(w1);
        pos += dim * hidden_dim + w1->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
        w2->set_group_size(group_size_);
        w2->set_weight(0, {dim, hidden_dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().w2_layers_.push_back(w2);
        pos += dim * hidden_dim + w2->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w3->set_group_size(group_size_);
        w3->set_weight(0, {hidden_dim, dim}, raw_model_data_->weight(pos), cpu_device_type);
        layers().w3_layers_.push_back(w3);
        pos += dim * hidden_dim + w3->get_scale_num() * sizeof(float);
    }

    auto cls_layer =
        std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
    cls_layer->set_group_size(group_size_);
    cls_layer->set_weight(0, {config_->vocab_size_, dim}, raw_model_data_->weight(pos),
                          cpu_device_type);
    if (!config_->is_shared_weight_) {
        pos += config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
    }
    layers().cls_layer_ = cls_layer;

    auto* weight_ptr = reinterpret_cast<float*>(const_cast<void*>(raw_model_data_->weight(pos)));
    layers().embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
    layers().embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                          cpu_device_type);
    weight_ptr += config_->vocab_size_ * dim;

    for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim,
                                                                 base::RmsNormEpsilon(model_type_));
        rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
        layers().rmsnorm_layers_.push_back(rms_norm_layer);
        weight_ptr += dim;
    }
    return base::error::Success();
}

void StandardDecoderModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    const auto& rmsnorm_layer = layers().rmsnorm_layers_.at(layer_idx);
    CHECK_NE(rmsnorm_layer, nullptr)
        << "The attention rmsnorm layer in the decoder model is null pointer.";
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void StandardDecoderModel::attention_qkv(int32_t layer_idx,
                                         const tensor::Tensor& pos_tensor) const {
    tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
    const int32_t pos = pos_tensor.index<int32_t>(0);
    auto [key, val] = slice_kv_cache(layer_idx, pos);
    auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);

    const auto& query_layer = layers().wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the decoder model is null pointer.";
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

    const auto& key_layer = layers().wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the decoder model is null pointer.";
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));

    const auto& value_layer = layers().wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the decoder model is null pointer.";
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

    apply_attention_projection_norms(layer_idx, query, key);

    CHECK_NE(layers().rope_layer_, nullptr)
        << "The RoPE layer in the decoder model is null pointer.";
    STATUS_CHECK(layers().rope_layer_->forward(
        query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

void StandardDecoderModel::attention_mha(int32_t layer_idx,
                                         const tensor::Tensor& pos_tensor) const {
    tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
    tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    tensor::Tensor query = get_buffer(ModelBufferType::kQuery);

    const auto& mha_layer = layers().mha_layer_;
    CHECK_NE(mha_layer, nullptr)
        << "The multi head attention layer in the decoder model is null pointer.";
    const int32_t pos = pos_tensor.index<int32_t>(0);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

    tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
    const auto& wo_layer = layers().wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr)
        << "The output projection layer in the decoder model is null pointer.";
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void StandardDecoderModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK_NE(layers().add_layer_, nullptr) << "The add layer in the decoder model is null pointer.";
    STATUS_CHECK(
        layers().add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

    tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = layers().rmsnorm_layers_.at(layer_idx + config_->layer_num_);
    CHECK_NE(ffn_rmsnorm, nullptr)
        << "The feedforward rmsnorm layer in the decoder model is null pointer.";
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

    tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = layers().w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the decoder model is null pointer.";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    const auto& w3_layer = layers().w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w3 layer in the decoder model is null pointer.";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

    CHECK_NE(layers().swiglu_layer_, nullptr)
        << "The swiglu layer in the decoder model is null pointer.";
    STATUS_CHECK(layers().swiglu_layer_->forward(w1_output, w3_output, w1_output));

    tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer = layers().w2_layers_.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the decoder model is null pointer.";
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

    STATUS_CHECK(layers().add_layer_->forward(input, w2_output, input));
}

void StandardDecoderModel::cls_logits(const tensor::Tensor& input) const {
    const auto& norm = layers().rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm, nullptr);
    STATUS_CHECK(norm->forward(input, input));

    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    CHECK_NE(layers().cls_layer_, nullptr);
    STATUS_CHECK(layers().cls_layer_->forward(input, forward_output));
}

int32_t StandardDecoderModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
    UNUSED(pos);
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    const float* forward_logits = forward_output.ptr<float>();

    if (is_prompt) {
        return -1;
    }
    return static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                 cuda_config_ ? cuda_config_->stream : nullptr));
}

}  // namespace model
