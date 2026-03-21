#include "model/llama/llama_decoder.h"
#include <glog/logging.h>
#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>
#include "base/topk_sampler.h"
#include "model/llama/llama_model_utils.h"
#include "op/kernels/cpu/rope_kernel.h"

namespace model {
namespace {
base::Status ValidateLayerGroup(const std::vector<std::shared_ptr<op::Layer>>& layers,
                                int32_t expected_size, const char* error_message) {
    if (static_cast<int32_t>(layers.size()) != expected_size) {
        return base::error::InternalError(error_message);
    }
    for (const auto& layer : layers) {
        if (!layer) {
            return base::error::InternalError(error_message);
        }
    }
    return base::error::Success();
}

base::Status ValidateOptionalLayerGroup(const std::vector<std::shared_ptr<op::Layer>>& layers,
                                        int32_t expected_size, const char* error_message) {
    if (layers.empty()) {
        return base::error::Success();
    }
    return ValidateLayerGroup(layers, expected_size, error_message);
}
}  // namespace

void LlamaDecoderLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
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

LlamaDecoderModel::LlamaDecoderModel(base::TokenizerType tokenizer_type,
                                           base::ModelType model_type, std::string token_path,
                                           std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, model_type, std::move(token_path), std::move(model_path),
            is_quant_model) {}

base::Status LlamaDecoderModel::init(base::DeviceType device_type,
                                        int32_t runtime_max_seq_len) {
    using namespace base;
    if (token_path_.empty()) {
        return error::PathNotValid(token_path_);
    }
    if (device_type == DeviceType::kDeviceCPU && is_quant_model_) {
        return error::InternalError("The cpu device do not support int8 quant model.");
    }
    if (runtime_max_seq_len != 0) {
        const auto max_seq_status = set_runtime_max_seq_len(runtime_max_seq_len);
        if (!max_seq_status.ok()) {
            return max_seq_status;
        }
    }

    device_type_ = device_type;
    if (device_type == DeviceType::kDeviceCUDA) {
        auto cuda_status = detail::InitCudaConfig(cuda_config_);
        if (!cuda_status.ok()) {
            return cuda_status;
        }
    }

    auto read_status = gen_model_from_file();
    if (!read_status.ok()) {
        return read_status;
    }

    init_mem();
    if (device_type_ == DeviceType::kDeviceCPU) {
        kernel::sin_cos_cache_calc_cpu(model_type_, config_->head_size_, config_->seq_len_,
                                       get_runtime_tensor(RuntimeTensorType::kSinCache).ptr<float>(),
                                       get_runtime_tensor(RuntimeTensorType::kCosCache).ptr<float>());
    } else {
        auto cache_status =
            detail::InitSinCosCache(model_type_, config_->head_size_, config_->seq_len_,
                                    get_runtime_tensor(RuntimeTensorType::kSinCache),
                                    get_runtime_tensor(RuntimeTensorType::kCosCache), cuda_config_);
        if (!cache_status.ok()) {
            return cache_status;
        }
    }

    if (!sampler_) {
        sampler_ = std::make_unique<sampler::TopKSampler>(device_type_);
    }
    return error::Success();
}

// 当前 token 的 embedding 
// 当前位置 tensor，里面存当前 step 的位置 pos
// 标记当前这一步是不是 prompt 阶段 true表示还在
// next：输出参数，用来保存预测出的下一个 token id
base::Status LlamaDecoderModel::predict(const tensor::Tensor& input,
                                           const tensor::Tensor& pos_tensor, bool is_prompt,
                                           int& next) const {
    // 输出不在这里 会写到 runtime tensor 里，主要是 kForwardOutput
    auto status = forward(input, pos_tensor, next);
    if (!status.ok()) {
        return status;
    }
    // 计算next, 为tokenid还是-1
    next = post_processing(pos_tensor, is_prompt);
    return base::error::Success();
}

base::Status LlamaDecoderModel::forward(const tensor::Tensor& input,
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
        // 输入 1xdim 输出 1xdim
        attention_rms(layer_idx, input);
        attention_qkv(layer_idx, pos_tensor);
        attention_mha(layer_idx, pos_tensor);
        feed_forward(layer_idx, input);
    }
    cls_logits(input);
    return base::error::Success();
}

op::EmbeddingOutput LlamaDecoderModel::embedding(const std::vector<int>& tokens) const {
    // Convert token ids into decoder input embeddings.
    auto input_tokens = get_runtime_tensor(RuntimeTensorType::kInputTokens);
    auto input_embeddings = get_runtime_tensor(RuntimeTensorType::kInputEmbeddings);
    const int32_t embedding_width = residual_width();
    if (input_tokens.size() != tokens.size()) {
        input_tokens.reshape({static_cast<int32_t>(tokens.size())});
        input_embeddings.reshape({static_cast<int32_t>(tokens.size()), embedding_width});
    }
    for (int32_t i = 0; i < static_cast<int32_t>(tokens.size()); ++i) {
        // 把 tokens 里的第 i 个 token id 写进 input_tokens 这个 Tensor
        input_tokens.index<int32_t>(i) = tokens.at(i);
    }

    LOG_IF(FATAL, !layers().embedding_layer_)
        << "The embedding layer in the decoder model is null pointer.";
    // 调用emb_kernel_cu_fp32 将tokenid对应的一维向量赋值给 该token所在索引位置
    STATUS_CHECK(layers().embedding_layer_->forward(input_tokens, input_embeddings));
    // 返回结果 包装了token id 的 Tensor embedding 结果的 Tensor token 数量
    return op::EmbeddingOutput(input_tokens, input_embeddings, static_cast<int32_t>(tokens.size()));
}

LlamaDecoderLayers& LlamaDecoderModel::layers() {
    CHECK(layers_ != nullptr) << "The decoder layers are null pointer.";
    return *layers_;
}

const LlamaDecoderLayers& LlamaDecoderModel::layers() const {
    CHECK(layers_ != nullptr) << "The decoder layers are null pointer.";
    return *layers_;
}

void LlamaDecoderModel::init_mem() {
    // Allocate runtime buffers for embeddings, intermediate activations, and the KV cache.
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
    CHECK(insert_runtime_tensor(RuntimeTensorType::kSinCache, sin_cache).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kCosCache, cos_cache).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kInputTokens, input_tokens).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kInputEmbeddings, input_embeddings).ok());
    tensor::Tensor rms_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    tensor::Tensor mha_output(base::DataType::kDataTypeFp32, attention_dim, true, alloc);
    tensor::Tensor w2_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    tensor::Tensor ffn_rms_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kOutputRMSNorm, rms_output).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kOutputMHA, mha_output).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kW2Output, w2_output).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kFFNRMSNorm, ffn_rms_output).ok());
    tensor::Tensor w1_output(base::DataType::kDataTypeFp32, ffn_dim, true, alloc);
    tensor::Tensor w3_output(base::DataType::kDataTypeFp32, ffn_dim, true, alloc);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kW1Output, w1_output).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kW3Output, w3_output).ok());

    tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                             config_->kv_dim_, true, alloc);
    tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_,
                               config_->seq_len_, config_->kv_dim_, true, alloc);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kKeyCache, key_cache).ok());
    CHECK(insert_runtime_tensor(RuntimeTensorType::kValueCache, value_cache).ok());
    tensor::Tensor query(base::DataType::kDataTypeFp32, attention_dim, true, alloc);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kQuery, query).ok());
    tensor::Tensor pos_tensor(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kInputPos, pos_tensor).ok());

    tensor::Tensor attn(base::DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true,
                        alloc);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kScoreStorage, attn).ok());
    tensor::Tensor attn_output(base::DataType::kDataTypeFp32, residual_dim, true, alloc);
    CHECK(insert_runtime_tensor(RuntimeTensorType::kAttnOutput, attn_output).ok());
    tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        tensor::Tensor forward_output_cpu(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                          alloc_cpu);
        CHECK(insert_runtime_tensor(RuntimeTensorType::kForwardOutputCPU, forward_output_cpu).ok());
    }
    CHECK(insert_runtime_tensor(RuntimeTensorType::kForwardOutput, forward_output).ok());
}

base::Status LlamaDecoderModel::create_layers() {
    using namespace base;
    if (!layers_) {
        layers_ = std::make_unique<LlamaDecoderLayers>();
    }

    base::Status layer_status =
        !is_quant_model_ ? create_param_layers() : create_param_quant_layers();
    if (!layer_status.ok()) {
        return layer_status;
    }

    layer_status = create_nonparam_layers();
    if (!layer_status.ok()) {
        return layer_status;
    }

    if (!layers().embedding_layer_) {
        return error::InternalError("Create the embedding layer for the decoder model failed!");
    }
    if (!layers().rope_layer_ || !layers().add_layer_ || !layers().mha_layer_ ||
        !layers().swiglu_layer_) {
        return error::InternalError(
            "Create the non-parameter layers for the decoder model "
            "failed.");
    }
    layer_status = ValidateLayerGroup(
        layers().rmsnorm_layers_, 2 * config_->layer_num_ + 1,
        "Create the rmsnorm layers for the decoder model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().wq_layers_, config_->layer_num_,
        "Create the attention layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().wk_layers_, config_->layer_num_,
        "Create the attention layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().wv_layers_, config_->layer_num_,
        "Create the attention layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().wo_layers_, config_->layer_num_,
        "Create the attention layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().w1_layers_, config_->layer_num_,
        "Create the feedforward layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().w2_layers_, config_->layer_num_,
        "Create the feedforward layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        layers().w3_layers_, config_->layer_num_,
        "Create the feedforward layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateOptionalLayerGroup(
        layers().query_norm_layers_, config_->layer_num_,
        "Create the query norm layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateOptionalLayerGroup(
        layers().key_norm_layers_, config_->layer_num_,
        "Create the key norm layers for the decoder model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }

    return validate_custom_layers();
}

base::Status LlamaDecoderModel::create_nonparam_layers() {
    layers().rope_layer_ = std::make_shared<op::RoPELayer>(device_type_, model_type_, config_->dim_,
                                                           config_->kv_dim_, config_->head_size_);

    layers().mha_layer_ = std::make_shared<op::MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
        config_->head_size_);
    layers().add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);
    layers().swiglu_layer_ = std::make_shared<op::SwiGLULayer>(device_type_, ffn_width());
    return base::error::Success();
}

base::Status LlamaDecoderModel::create_param_quant_layers() {
    CHECK(is_quant_model_);
    size_t pos = 0;
    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const auto cpu_device_type = base::DeviceType::kDeviceCPU;
    const auto set_weight_or_die =
        [&](const auto& layer, const std::vector<int32_t>& dims, const void* weight_ptr) {
            const auto status = layer->set_weight(0, dims, weight_ptr, cpu_device_type);
            CHECK(status.ok()) << status.message();
        };

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
        wq->set_group_size(group_size_);
        set_weight_or_die(wq, {dim, dim}, raw_model_data_->weight(pos));
        layers().wq_layers_.push_back(wq);
        pos += dim * dim + wq->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wk = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
        wk->set_group_size(group_size_);
        set_weight_or_die(wk, {config_->kv_dim_, dim}, raw_model_data_->weight(pos));
        layers().wk_layers_.push_back(wk);
        pos += config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wv = std::make_shared<op::MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
        wv->set_group_size(group_size_);
        set_weight_or_die(wv, {config_->kv_dim_, dim}, raw_model_data_->weight(pos));
        layers().wv_layers_.push_back(wv);
        pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wo = std::make_shared<op::MatmulLayer>(device_type_, dim, dim, true);
        wo->set_group_size(group_size_);
        set_weight_or_die(wo, {dim, dim}, raw_model_data_->weight(pos));
        layers().wo_layers_.push_back(wo);
        pos += dim * dim + wo->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w1 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w1->set_group_size(group_size_);
        set_weight_or_die(w1, {hidden_dim, dim}, raw_model_data_->weight(pos));
        layers().w1_layers_.push_back(w1);
        pos += dim * hidden_dim + w1->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w2 = std::make_shared<op::MatmulLayer>(device_type_, dim, hidden_dim, true);
        w2->set_group_size(group_size_);
        set_weight_or_die(w2, {dim, hidden_dim}, raw_model_data_->weight(pos));
        layers().w2_layers_.push_back(w2);
        pos += dim * hidden_dim + w2->get_scale_num() * sizeof(float);
    }

    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto w3 = std::make_shared<op::MatmulLayer>(device_type_, hidden_dim, dim, true);
        w3->set_group_size(group_size_);
        set_weight_or_die(w3, {hidden_dim, dim}, raw_model_data_->weight(pos));
        layers().w3_layers_.push_back(w3);
        pos += dim * hidden_dim + w3->get_scale_num() * sizeof(float);
    }

    const auto cls_and_embedding = detail::ResolveLegacyQuantizedWeightsLayout(
        *raw_model_data_, pos, config_->vocab_size_, dim, group_size_, config_->is_shared_weight_);
    std::shared_ptr<op::MatmulLayer> cls_layer;
    if (cls_and_embedding.classifier_is_quantized) {
        cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
        cls_layer->set_group_size(group_size_);
        set_weight_or_die(cls_layer, {config_->vocab_size_, dim}, cls_and_embedding.classifier_weight);
        pos += detail::LegacyQuantizedTensorBytes(config_->vocab_size_, dim, group_size_);
    } else {
        cls_layer = std::make_shared<op::MatmulLayer>(device_type_, config_->vocab_size_, dim);
        set_weight_or_die(cls_layer, {config_->vocab_size_, dim}, cls_and_embedding.classifier_weight);
    }
    layers().cls_layer_ = cls_layer;
    auto* weight_ptr =
        reinterpret_cast<float*>(const_cast<void*>(cls_and_embedding.embedding_weight));
    layers().embedding_layer_ = std::make_shared<op::EmbeddingLayer>(
        device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
    set_weight_or_die(layers().embedding_layer_, {std::abs(config_->vocab_size_), dim}, weight_ptr);
    weight_ptr += config_->vocab_size_ * dim;

    for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i) {
        auto rms_norm_layer = std::make_shared<op::RmsNormLayer>(device_type_, dim,
                                                                 base::RmsNormEpsilon(model_type_));
        set_weight_or_die(rms_norm_layer, {dim}, weight_ptr);
        layers().rmsnorm_layers_.push_back(rms_norm_layer);
        weight_ptr += dim;
    }
    return base::error::Success();
}

void LlamaDecoderModel::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
    // 获取上一步的rmsnorm的结果
    tensor::Tensor rmsnorm_output = get_runtime_tensor(RuntimeTensorType::kOutputRMSNorm);
    const auto& rmsnorm_layer = layers().rmsnorm_layers_.at(layer_idx);
    CHECK_NE(rmsnorm_layer, nullptr)
        << "The attention rmsnorm layer in the decoder model is null pointer.";
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LlamaDecoderModel::attention_qkv(int32_t layer_idx,
                                         const tensor::Tensor& pos_tensor) const {
    // 运行时缓存里取出 kQuery 这个 Tensor，作为当前层 attention 里保存 Q 向量的缓冲区
    // 1 x attention_dim  attention_dim = config_->dim_
    tensor::Tensor query = get_runtime_tensor(RuntimeTensorType::kQuery);
    const int32_t pos = pos_tensor.index<int32_t>(0);
    // 从整块 key/value cache 里，切出“当前层、当前位置”对应的那一小段 K 和 V 缓冲区返回
    // 1 x kv_dim 1 x kv_dim
    auto [key, val] = slice_kv_cache(layer_idx, pos);
    // 取出output的结果
    auto rmsnorm_output = get_runtime_tensor(RuntimeTensorType::kOutputRMSNorm);
    // 从当前层的 wq_layers_ 里取出第 layer_idx 层对应的 Wq 线性层
    const auto& query_layer = layers().wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the decoder model is null pointer.";
    // 1 x attention_dim | [attention_dim, dim] * [dim] 保存在query
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
    const auto& key_layer = layers().wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the decoder model is null pointer.";
    // 1 x kv_dim | [kv_dim, dim] * [dim]
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
    const auto& value_layer = layers().wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the decoder model is null pointer.";
    // 1 x kv_dim | [kv_dim, dim] * [dim]
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));
    // hook 现在啥也不做
    apply_attention_projection_norms(layer_idx, query, key);
    CHECK_NE(layers().rope_layer_, nullptr)
        << "The RoPE layer in the decoder model is null pointer.";
    tensor::Tensor sin_cache = get_runtime_tensor(RuntimeTensorType::kSinCache);
    tensor::Tensor cos_cache = get_runtime_tensor(RuntimeTensorType::kCosCache);
    // RoPE
    STATUS_CHECK(layers().rope_layer_->forward(query, key, pos_tensor, sin_cache, cos_cache,
                                               tensor::Tensor{}));
}

void LlamaDecoderModel::attention_mha(int32_t layer_idx,
                                         const tensor::Tensor& pos_tensor) const {
    tensor::Tensor key_cache = get_runtime_tensor(RuntimeTensorType::kKeyCache);
    tensor::Tensor val_cache = get_runtime_tensor(RuntimeTensorType::kValueCache);
    tensor::Tensor mha_output = get_runtime_tensor(RuntimeTensorType::kOutputMHA);
    tensor::Tensor score_storage = get_runtime_tensor(RuntimeTensorType::kScoreStorage);
    tensor::Tensor query = get_runtime_tensor(RuntimeTensorType::kQuery);
    const auto& mha_layer = layers().mha_layer_;
    CHECK_NE(mha_layer, nullptr)
        << "The multi head attention layer in the decoder model is null pointer.";
    const int32_t pos = pos_tensor.index<int32_t>(0);
    // mha_layer 静态类型是 shared_ptr<op::Layer>，
    // 要调用 set_pos 和 set_layer_idx 这些 MultiHeadAttention 特有函数，
    // 必须先转成 MultiHeadAttention
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_pos(pos);
    std::dynamic_pointer_cast<op::MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
    tensor::Tensor attn_output = get_runtime_tensor(RuntimeTensorType::kAttnOutput);
    const auto& wo_layer = layers().wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr)
        << "The output projection layer in the decoder model is null pointer.";
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LlamaDecoderModel::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK_NE(layers().add_layer_, nullptr) << "The add layer in the decoder model is null pointer.";
    STATUS_CHECK(
        layers().add_layer_->forward(input, get_runtime_tensor(RuntimeTensorType::kAttnOutput), input));
    tensor::Tensor ffn_norm_output = get_runtime_tensor(RuntimeTensorType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = layers().rmsnorm_layers_.at(layer_idx + config_->layer_num_);
    CHECK_NE(ffn_rmsnorm, nullptr)
        << "The feedforward rmsnorm layer in the decoder model is null pointer.";
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
    tensor::Tensor w1_output = get_runtime_tensor(RuntimeTensorType::kW1Output);
    const auto& w1_layer = layers().w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the decoder model is null pointer.";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    tensor::Tensor w3_output = get_runtime_tensor(RuntimeTensorType::kW3Output);
    const auto& w3_layer = layers().w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w3 layer in the decoder model is null pointer.";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

    CHECK_NE(layers().swiglu_layer_, nullptr)
        << "The swiglu layer in the decoder model is null pointer.";
    STATUS_CHECK(layers().swiglu_layer_->forward(w1_output, w3_output, w1_output));
    tensor::Tensor w2_output = get_runtime_tensor(RuntimeTensorType::kW2Output);
    const auto& w2_layer = layers().w2_layers_.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the decoder model is null pointer.";
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));
    // 结果还是写回input了
    STATUS_CHECK(layers().add_layer_->forward(input, w2_output, input));
}

void LlamaDecoderModel::cls_logits(const tensor::Tensor& input) const {
    const auto& norm = layers().rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm, nullptr);
    STATUS_CHECK(norm->forward(input, input));
    tensor::Tensor forward_output = get_runtime_tensor(RuntimeTensorType::kForwardOutput);
    CHECK_NE(layers().cls_layer_, nullptr);
    // 结果forward_output 也就是kForwardOutput
    STATUS_CHECK(layers().cls_layer_->forward(input, forward_output));
}

int32_t LlamaDecoderModel::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
    UNUSED(pos);
    // 如果当前还是 prompt 阶段，直接返回 -1
    if (is_prompt) {
        return -1;
    }
    CHECK_NE(sampler_, nullptr) << "The sampler is null.";

    // 从 kForwardOutput 里取出 logits
    const tensor::Tensor& forward_output = get_runtime_tensor(RuntimeTensorType::kForwardOutput);
    const tensor::Tensor* sampling_output = &forward_output;
    if (sampler_->requires_host_logits(forward_output.device_type())) {
        const tensor::Tensor& forward_output_cpu =
            get_runtime_tensor(RuntimeTensorType::kForwardOutputCPU);
        CHECK_EQ(forward_output_cpu.size(), forward_output.size())
            << "The CPU logits buffer size does not match the forward output.";
        forward_output_cpu.get_runtime_tensor()->copy_from(forward_output.get_runtime_tensor().get());
        sampling_output = &forward_output_cpu;
    }

    void* sample_stream = nullptr;
    if (sampling_output->device_type() == base::DeviceType::kDeviceCUDA && cuda_config_) {
        sample_stream = cuda_config_->stream;
    }

    // 如果已经进入生成阶段，就用 sampler_->sample(...) 从 logits 里选出下一个 token id
    return static_cast<int32_t>(
        sampler_->sample(sampling_output->ptr<float>(), sampling_output->size(), sample_stream));
}
}  // namespace model
