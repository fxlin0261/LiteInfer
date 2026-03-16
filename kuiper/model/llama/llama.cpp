#include "model/llama/llama.h"

#include <glog/logging.h>

#include <cstdlib>
#include <utility>

#include <op/matmul.h>
#include <op/mha.h>
#include <op/rmsnorm.h>

#include "model/decoder/model_utils.h"
#include "op/kernels/cpu/rope_kernel.h"

namespace model {
namespace {
using LayerPtr = std::shared_ptr<op::Layer>;
using LayerList = std::vector<LayerPtr>;

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

void ReserveLayerStorage(LlamaLayers& layers, int32_t layer_num) {
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
    layer->set_weight(0, {out_features, in_features}, weight_ptr, weight_device);
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
    layer->set_weight(0, {out_features, in_features}, weight_ptr, weight_device);
    return layer;
}

std::shared_ptr<op::EmbeddingLayer> CreateEmbeddingLayer(base::DeviceType device_type, int32_t dim,
                                                         int32_t seq_len, int32_t vocab_size,
                                                         const void* weight_ptr,
                                                         base::DeviceType weight_device) {
    auto layer = std::make_shared<op::EmbeddingLayer>(device_type, dim, seq_len, vocab_size);
    layer->set_weight(0, {vocab_size, dim}, weight_ptr, weight_device);
    return layer;
}

std::shared_ptr<op::RmsNormLayer> CreateRmsNormLayer(base::DeviceType device_type,
                                                     base::ModelType model_type, int32_t dim,
                                                     const void* weight_ptr,
                                                     base::DeviceType weight_device) {
    auto layer =
        std::make_shared<op::RmsNormLayer>(device_type, dim, base::RmsNormEpsilon(model_type));
    layer->set_weight(0, {dim}, weight_ptr, weight_device);
    return layer;
}

size_t AppendQuantizedMatmulLayers(const RawModelData& raw_model_data, int32_t layer_num,
                                   int32_t out_features, int32_t in_features, size_t offset,
                                   int32_t group_size, base::DeviceType device_type,
                                   base::DeviceType weight_device, LayerList& layers) {
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

base::Status ValidateRequiredLayer(const LayerPtr& layer, const char* error_message) {
    if (!layer) {
        return base::error::InternalError(error_message);
    }
    return base::error::Success();
}

base::Status ValidateLayerGroup(const LayerList& layers, int32_t expected_size,
                                const char* error_message) {
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

op::MultiHeadAttention& CheckedMhaLayer(const LayerPtr& layer) {
    auto mha_layer = std::dynamic_pointer_cast<op::MultiHeadAttention>(layer);
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer has an unexpected type.";
    return *mha_layer;
}
}  // namespace

void LlamaLayers::to_cuda(std::shared_ptr<kernel::CudaConfig> config) {
    detail::MoveLayerToCuda(add_layer_, config);
    detail::MoveLayerToCuda(rope_layer_, config);
    detail::MoveLayerToCuda(swiglu_layer_, config);
    detail::MoveLayerToCuda(mha_layer_, config);
    detail::MoveLayerToCuda(cls_layer_, config);
    detail::MoveLayerToCuda(embedding_layer_, config);

    detail::MoveLayerRangeToCuda(wq_layers_, config);
    detail::MoveLayerRangeToCuda(wk_layers_, config);
    detail::MoveLayerRangeToCuda(wv_layers_, config);
    detail::MoveLayerRangeToCuda(wo_layers_, config);
    detail::MoveLayerRangeToCuda(w1_layers_, config);
    detail::MoveLayerRangeToCuda(w2_layers_, config);
    detail::MoveLayerRangeToCuda(w3_layers_, config);
    detail::MoveLayerRangeToCuda(rmsnorm_layers_, config);
}

LlamaModelBase::LlamaModelBase(base::TokenizerType tokenizer_type, base::ModelType model_type,
                               std::string token_path, std::string model_path, bool is_quant_model)
    : Model(tokenizer_type, model_type, std::move(token_path), std::move(model_path),
            is_quant_model) {}

base::Status LlamaModelBase::init(base::DeviceType device_type) {
    using namespace base;
    if (token_path_.empty()) {
        return error::PathNotValid(token_path_);
    }
    // 这个项目里 CPU 不支持 int8 量化模型推理
    if (device_type == DeviceType::kDeviceCPU && is_quant_model_) {
        return error::InternalError("The cpu device do not support int8 quant model.");
    }

    device_type_ = device_type;
    if (device_type == DeviceType::kDeviceCUDA) {
        const auto cuda_status = detail::InitCudaConfig(cuda_config_);
        if (!cuda_status.ok()) {
            return cuda_status;
        }
    }
    // 这一步是“正式加载模型”的入口。它会从模型文件里解析配置、权重，并创建各层对象
    const Status read_status = gen_model_from_file();
    if (!read_status.ok()) {
        return read_status;
    }
    // 这一步是分配运行时缓冲区，不是读权重。权重是模型参数，已经由前面的 gen_model_from_file() 处理
    // 而 init_mem() 更像是在给推理过程准备工作区，比如中间结果、KV cache、logits buffer、sin/cos
    // cache 这些
    init_mem();
    // 是在预计算 RoPE 位置编码 需要的 sin / cos 表。
    // 原因是 Transformer 在每个 token、每个 head 上都要用到旋转位置编码，
    // 如果每次推理时临时算三角函数会很慢，所以初始化阶段直接按
    // RoPE 层运行前，它依赖的 sin/cos 数据已经准备好了
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        kernel::sin_cos_cache_calc_cpu(model_type_, config_->head_size_, config_->seq_len_,
                                       get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                       get_buffer(ModelBufferType::kCosCache).ptr<float>());
    } else {
        const auto cache_status =
            detail::InitSinCosCache(model_type_, config_->head_size_, config_->seq_len_,
                                    get_buffer(ModelBufferType::kSinCache),
                                    get_buffer(ModelBufferType::kCosCache), cuda_config_);
        if (!cache_status.ok()) {
            return cache_status;
        }
    }
    // 这里创建的是采样器。当前实现用的是 ArgmaxSampler，也就是每次从 logits 里直接选概率最大的
    // token。 所以这个项目目前默认不是 top-k / top-p 随机采样，而是更确定性的 greedy decoding
    sampler_ = std::make_unique<sampler::ArgmaxSampler>(device_type_);
    return error::Success();
}

base::Status LlamaModelBase::forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                     int& next) const {
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

base::Status LlamaModelBase::create_nonparam_layers() {
    CHECK(llama_layers_ != nullptr);
    llama_layers_->rope_layer_ = std::make_shared<op::RoPELayer>(
        device_type_, model_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

    llama_layers_->mha_layer_ = std::make_shared<op::MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
        config_->head_size_);
    llama_layers_->add_layer_ = std::make_shared<op::VecAddLayer>(device_type_);

    llama_layers_->swiglu_layer_ =
        std::make_shared<op::SwiGLULayer>(device_type_, config_->hidden_dim_);
    return base::error::Success();
}

base::Status LlamaModelBase::create_param_quant_layers() {
    CHECK(is_quant_model_);
    CHECK(llama_layers_ != nullptr);
    const auto weight_device = base::DeviceType::kDeviceCPU;
    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const int32_t layer_num = config_->layer_num_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t vocab_size = std::abs(config_->vocab_size_);

    size_t offset = 0;
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->wq_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, kv_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->wk_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, kv_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->wv_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->wo_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, hidden_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->w1_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, dim, hidden_dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->w2_layers_);
    offset = AppendQuantizedMatmulLayers(*raw_model_data_, layer_num, hidden_dim, dim, offset,
                                         group_size_, device_type_, weight_device,
                                         llama_layers_->w3_layers_);

    const auto cls_and_embedding = detail::ResolveLegacyQuantizedWeightsLayout(
        *raw_model_data_, offset, vocab_size, dim, group_size_, config_->is_shared_weight_);
    if (cls_and_embedding.classifier_is_quantized) {
        llama_layers_->cls_layer_ = CreateQuantizedMatmulLayer(
            device_type_, vocab_size, dim, cls_and_embedding.classifier_weight, weight_device,
            group_size_);
    } else {
        llama_layers_->cls_layer_ = CreateMatmulLayer(device_type_, vocab_size, dim,
                                                      cls_and_embedding.classifier_weight,
                                                      weight_device);
    }

    auto* weight_cursor =
        reinterpret_cast<float*>(const_cast<void*>(cls_and_embedding.embedding_weight));
    llama_layers_->embedding_layer_ =
        CreateEmbeddingLayer(device_type_, dim, config_->seq_len_, vocab_size, weight_cursor,
                             weight_device);
    weight_cursor += static_cast<size_t>(vocab_size) * dim;

    const int32_t rms_layer_count = 2 * layer_num + 1;
    for (int32_t layer_idx = 0; layer_idx < rms_layer_count; ++layer_idx) {
        llama_layers_->rmsnorm_layers_.push_back(
            CreateRmsNormLayer(device_type_, model_type_, dim, weight_cursor, weight_device));
        weight_cursor += dim;
    }
    return base::error::Success();
}

base::Status LlamaModelBase::create_param_layers() {
    CHECK(!is_quant_model_);
    CHECK(llama_layers_ != nullptr);
    const auto weight_device = base::DeviceType::kDeviceCPU;
    const int32_t dim = config_->dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const int32_t layer_num = config_->layer_num_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t vocab_size = std::abs(config_->vocab_size_);
    const DenseWeightsLayout layout = BuildDenseWeightsLayout(*config_);

    llama_layers_->embedding_layer_ = CreateEmbeddingLayer(
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

        llama_layers_->rmsnorm_layers_.push_back(CreateRmsNormLayer(
            device_type_, model_type_, dim, raw_model_data_->weight(attn_rms_offset),
            weight_device));
        llama_layers_->wq_layers_.push_back(
            CreateMatmulLayer(device_type_, dim, dim, raw_model_data_->weight(wq_offset),
                              weight_device));
        llama_layers_->wk_layers_.push_back(
            CreateMatmulLayer(device_type_, kv_dim, dim, raw_model_data_->weight(wk_offset),
                              weight_device));
        llama_layers_->wv_layers_.push_back(
            CreateMatmulLayer(device_type_, kv_dim, dim, raw_model_data_->weight(wv_offset),
                              weight_device));
        llama_layers_->wo_layers_.push_back(
            CreateMatmulLayer(device_type_, dim, dim, raw_model_data_->weight(wo_offset),
                              weight_device));
        llama_layers_->rmsnorm_layers_.push_back(CreateRmsNormLayer(
            device_type_, model_type_, dim, raw_model_data_->weight(ffn_rms_offset),
            weight_device));
        llama_layers_->w1_layers_.push_back(
            CreateMatmulLayer(device_type_, hidden_dim, dim, raw_model_data_->weight(w1_offset),
                              weight_device));
        llama_layers_->w2_layers_.push_back(CreateMatmulLayer(
            device_type_, dim, hidden_dim, raw_model_data_->weight(w2_offset), weight_device));
        llama_layers_->w3_layers_.push_back(
            CreateMatmulLayer(device_type_, hidden_dim, dim, raw_model_data_->weight(w3_offset),
                              weight_device));
    }

    llama_layers_->rmsnorm_layers_.push_back(CreateRmsNormLayer(
        device_type_, model_type_, dim, raw_model_data_->weight(layout.final_rms), weight_device));

    const void* cls_weight = config_->is_shared_weight_ ? raw_model_data_->weight(layout.embedding)
                                                        : raw_model_data_->weight(layout.cls);
    llama_layers_->cls_layer_ =
        CreateMatmulLayer(device_type_, vocab_size, dim, cls_weight, weight_device);
    return base::error::Success();
}

void LlamaModelBase::init_mem() {
    CHECK(llama_layers_ != nullptr);

    const auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
    const auto device_alloc = device_type_ == base::DeviceType::kDeviceCPU
                                  ? cpu_alloc
                                  : base::CUDADeviceAllocatorFactory::get_instance();

    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        CHECK_NE(cuda_config_, nullptr);
        llama_layers_->to_cuda(cuda_config_);
    }

    const auto register_buffer = [&](ModelBufferType buffer_type, const tensor::Tensor& tensor) {
        CHECK(insert_buffer(buffer_type, tensor).ok());
    };

    register_buffer(ModelBufferType::kInputTokens,
                    tensor::Tensor(base::DataType::kDataTypeInt32, 1, true, cpu_alloc));
    register_buffer(ModelBufferType::kInputEmbeddings,
                    tensor::Tensor(base::DataType::kDataTypeFp32, 1, config_->dim_, true,
                                   device_alloc));
    register_buffer(ModelBufferType::kSinCache,
                    tensor::Tensor(base::DataType::kDataTypeFp32,
                                   config_->head_size_ * config_->seq_len_, true, device_alloc));
    register_buffer(ModelBufferType::kCosCache,
                    tensor::Tensor(base::DataType::kDataTypeFp32,
                                   config_->head_size_ * config_->seq_len_, true, device_alloc));

    // These activations are used in different stages of one decode step, so they intentionally
    // share scratch storage through Tensor's shallow-copy semantics.
    tensor::Tensor shared_dim_scratch(base::DataType::kDataTypeFp32, config_->dim_, true,
                                      device_alloc);
    register_buffer(ModelBufferType::kOutputRMSNorm, shared_dim_scratch);
    register_buffer(ModelBufferType::kOutputMHA, shared_dim_scratch);
    register_buffer(ModelBufferType::kW2Output, shared_dim_scratch);
    register_buffer(ModelBufferType::kFFNRMSNorm, shared_dim_scratch);

    tensor::Tensor shared_hidden_scratch(base::DataType::kDataTypeFp32, config_->hidden_dim_, true,
                                         device_alloc);
    register_buffer(ModelBufferType::kW1Output, shared_hidden_scratch);
    register_buffer(ModelBufferType::kW3Output, shared_hidden_scratch);

    register_buffer(ModelBufferType::kKeyCache,
                    tensor::Tensor(base::DataType::kDataTypeFp32, config_->layer_num_,
                                   config_->seq_len_, config_->kv_dim_, true, device_alloc));
    register_buffer(ModelBufferType::kValueCache,
                    tensor::Tensor(base::DataType::kDataTypeFp32, config_->layer_num_,
                                   config_->seq_len_, config_->kv_dim_, true, device_alloc));

    tensor::Tensor query_and_attn_output(base::DataType::kDataTypeFp32, config_->dim_, true,
                                         device_alloc);
    register_buffer(ModelBufferType::kQuery, query_and_attn_output);
    register_buffer(ModelBufferType::kAttnOutput, query_and_attn_output);

    register_buffer(ModelBufferType::kInputPos,
                    tensor::Tensor(base::DataType::kDataTypeInt32, 1, true, cpu_alloc));
    register_buffer(ModelBufferType::kScoreStorage,
                    tensor::Tensor(base::DataType::kDataTypeFp32, config_->head_num_,
                                   config_->seq_len_, true, device_alloc));

    tensor::Tensor forward_output(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                  device_alloc);
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
        register_buffer(ModelBufferType::kForwardOutputCPU,
                        tensor::Tensor(base::DataType::kDataTypeFp32, config_->vocab_size_, true,
                                       cpu_alloc));
    }

    register_buffer(ModelBufferType::kForwardOutput, forward_output);
}

base::Status LlamaModelBase::create_layers() {
    using namespace base;
    llama_layers_ = std::make_unique<LlamaLayers>();
    ReserveLayerStorage(*llama_layers_, config_->layer_num_);

    Status layer_status = is_quant_model_ ? create_param_quant_layers() : create_param_layers();
    if (!layer_status.ok()) {
        return layer_status;
    }

    layer_status = create_nonparam_layers();
    if (!layer_status.ok()) {
        return layer_status;
    }

    layer_status = ValidateRequiredLayer(
        llama_layers_->embedding_layer_,
        "Create the embedding layer for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateRequiredLayer(llama_layers_->cls_layer_,
                                         "Create the cls layer for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->rmsnorm_layers_, 2 * config_->layer_num_ + 1,
        "Create the rmsnorm layers for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->wq_layers_, config_->layer_num_,
        "Create the query matmul layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->wk_layers_, config_->layer_num_,
        "Create the key matmul layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->wv_layers_, config_->layer_num_,
        "Create the value matmul layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->wo_layers_, config_->layer_num_,
        "Create the output projection layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->w1_layers_, config_->layer_num_,
        "Create the w1 matmul layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->w2_layers_, config_->layer_num_,
        "Create the w2 matmul layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateLayerGroup(
        llama_layers_->w3_layers_, config_->layer_num_,
        "Create the w3 matmul layers for the Llama model failed.");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateRequiredLayer(llama_layers_->rope_layer_,
                                         "Create the rope layer for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateRequiredLayer(llama_layers_->add_layer_,
                                         "Create the add layer for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateRequiredLayer(llama_layers_->mha_layer_,
                                         "Create the mha layer for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    layer_status = ValidateRequiredLayer(llama_layers_->swiglu_layer_,
                                         "Create the SwiGLU layer for the Llama model failed!");
    if (!layer_status.ok()) {
        return layer_status;
    }
    return error::Success();
}

op::EmbeddingOutput LlamaModelBase::embedding(const std::vector<int>& tokens) const {
    // 将 token id 映射成输入 embedding。
    auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
    auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
    const int32_t token_count = static_cast<int32_t>(tokens.size());
    if (input_tokens.size() != tokens.size()) {
        input_tokens.reshape({token_count});
        input_embeddings.reshape({token_count, config_->dim_});
    }
    for (int32_t token_idx = 0; token_idx < token_count; ++token_idx) {
        input_tokens.index<int32_t>(token_idx) = tokens.at(token_idx);
    }
    LOG_IF(FATAL, !llama_layers_->embedding_layer_)
        << "The embedding layer in the Llama model is null pointer.";
    STATUS_CHECK(llama_layers_->embedding_layer_->forward(input_tokens, input_embeddings));
    return op::EmbeddingOutput(input_tokens, input_embeddings, token_count);
}

void LlamaModelBase::attention_rms(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK(llama_layers_ != nullptr);
    // 注意力块前的 RMSNorm。
    // 从模型内部拿一个缓冲区出来, 名字是 kOutputRMSNorm, 这个缓冲区用来存放 RMSNorm 的输出结果
    tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    // 从 llama_layers_->rmsnorm_layers_ 这个数组/容器里，取出第 layer_idx 层对应的 RMSNorm 层对象。
    const auto& rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
    if (!rmsnorm_layer) {
        LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the Llama model";
    }
    // 调用这一层的 forward: 输入是 input 输出写到 rmsnorm_output
    STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
}

void LlamaModelBase::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    CHECK(llama_layers_ != nullptr);
    // 计算当前层的 Q/K/V，并对 Q/K 应用 RoPE。
    // 从内部 buffer 里拿一块空间，准备存当前层算出来的Q
    tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
    // pos：从 pos_tensor 里读出当前位置整数
    const int32_t pos = pos_tensor.index<int32_t>(0);
    // 从 KV cache 里切出当前层、当前位置对应的 K 和 V 存储位置
    const auto [key, val] = slice_kv_cache(layer_idx, pos);
    // 也就是说：Q 存在临时 query buffer, K/V 直接写进 cache 对应槽位

    // 接着取出这层的 Wq：query_layer 本质上就是线性层 Wq
    const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
    CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
    // 再拿 RMSNorm 输出：
    const tensor::Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
    // 也就是用 attention 前 RMSNorm 的结果来生成 query
    // Q = Wq * rmsnorm_output
    STATUS_CHECK(query_layer->forward(rmsnorm_output, query));
    // K = Wk * rmsnorm_output
    const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
    CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
    STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
    // V = Wv * rmsnorm_output
    const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
    CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
    STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

    // RoPE 只改位置编码，不改变张量形状。
    // Q, K = RoPE(Q, K, pos)
    CHECK_NE(llama_layers_->rope_layer_, nullptr)
        << "The RoPE layer in the attention block is null pointer.";
    STATUS_CHECK(llama_layers_->rope_layer_->forward(
        query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
        get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}

base::Status LlamaModelBase::predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                                     bool is_prompt, int& next) const {
    // 执行一步解码；非 prompt 阶段会采样下一个 token。
    const auto status = forward(input, pos_tensor, next);
    if (!status.ok()) {
        return status;
    }
    next = post_processing(pos_tensor, is_prompt);
    return base::error::Success();
}

void LlamaModelBase::attention_mha(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
    CHECK(llama_layers_ != nullptr);
    // 用当前 query 和 KV cache 计算注意力输出。
    // key_cache：所有历史 token 的 K 缓存
    tensor::Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    // val_cache：所有历史 token 的 V 缓存
    tensor::Tensor val_cache = get_buffer(ModelBufferType::kValueCache);
    // mha_output：存多头注意力结果的输出缓冲区
    tensor::Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    // score_storage：存 attention 分数的中间缓冲区
    tensor::Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
    // query：当前 token 的 Q
    tensor::Tensor query = get_buffer(ModelBufferType::kQuery);
    // 然后取出 MHA 层对象：
    const auto& mha_layer = llama_layers_->mha_layer_;
    CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
    // 接着读取当前位置：
    auto& typed_mha_layer = CheckedMhaLayer(mha_layer);
    // 然后把当前位置和层号设置进 MultiHeadAttention 层：
    typed_mha_layer.set_pos(pos_tensor.index<int32_t>(0));
    typed_mha_layer.set_layer_idx(layer_idx);
    // 执行mha
    // scores = Q * K^T / sqrt(d)
    // scores = softmax(scores)
    // mha_output = scores * V
    STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));
    // 这里做的是 attention 输出投影
    tensor::Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
    // attn_output = Wo * mha_output
    const auto& wo_layer = llama_layers_->wo_layers_.at(layer_idx);
    CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
    STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
}

void LlamaModelBase::feed_forward(int32_t layer_idx, const tensor::Tensor& input) const {
    CHECK(llama_layers_ != nullptr);
    // 执行 FFN，并把结果写回残差流。
    CHECK_NE(llama_layers_->add_layer_, nullptr)
        << "The add layer in the feedforward block is null pointer";
    // input = input + attn_output
    STATUS_CHECK(
        llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));
    // ffn_norm_output = RMSNorm(input)
    tensor::Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
    const auto& ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
    CHECK_NE(ffn_rmsnorm, nullptr)
        << "The final rmsnorm layer in the feedforward block is null pointer";
    STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));
    // w1_output = W1(ffn_norm_output)
    // w3_output = W3(ffn_norm_output)
    tensor::Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
    const auto& w1_layer = llama_layers_->w1_layers_.at(layer_idx);
    CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
    STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));
    tensor::Tensor w3_output = get_buffer(ModelBufferType::kW3Output);
    const auto& w3_layer = llama_layers_->w3_layers_.at(layer_idx);
    CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
    STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_output));

    // SwiGLU 门控激活。
    // w1_output = SiLU(w1_output) * w3_output
    CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
        << "The swiglu layer in the feedforward block is null pointer";
    STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_output, w1_output));
    // w2_output = W2(w1_output)
    tensor::Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
    const auto& w2_layer = llama_layers_->w2_layers_.at(layer_idx);
    CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
    STATUS_CHECK(w2_layer->forward(w1_output, w2_output));
    CHECK_NE(llama_layers_->add_layer_, nullptr)
        << "The add layer in the feedforward block is null pointer";
    STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
}

void LlamaModelBase::cls_logits(const tensor::Tensor& input) const {
    CHECK(llama_layers_ != nullptr);
    // 最后一层归一化后投影到词表 logits。
    // input = RMSNorm(input)
    const auto& norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    CHECK_NE(norm, nullptr);
    STATUS_CHECK(norm->forward(input, input));
    // logits = W_vocab * input
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    CHECK_NE(llama_layers_->cls_layer_, nullptr);
    STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
}

int32_t LlamaModelBase::post_processing(const tensor::Tensor& pos, bool is_prompt) const {
    UNUSED(pos);
    tensor::Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
    const float* forward_logits = forward_output.ptr<float>();
    // 如果当前是在 prompt 阶段 可能不真正采样，只是把 prompt token 推进 KV cache
    if (is_prompt) {
        return -1;
    }
        // 如果当前已经进入生成阶段
        // 就会从 logits 里采样或取 argmax，得到下一个 token
        // forward() 负责把 logits 算出来，
        // post_processing() 负责把 logits 变成最终的 next token。
    return static_cast<int32_t>(
        sampler_->sample(forward_logits, forward_output.size(),
                         cuda_config_ ? cuda_config_->stream : nullptr));
}
}  // namespace model
