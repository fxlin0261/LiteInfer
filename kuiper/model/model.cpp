#include "model/model.h"
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <sys/mman.h>
#include <sys/stat.h>
namespace model {
Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::ModelType Model::model_type() const { return model_type_; }

const std::string& Model::token_path() const { return token_path_; }

const std::string& Model::model_path() const { return model_path_; }

base::Status Model::insert_buffer(ModelBufferType buffer_idx, const tensor::Tensor& tensor) {
    if (buffers_.count(buffer_idx) > 0) {
        return base::error::KeyHasExits(std::to_string(int(buffer_idx)) +
                                        " has exits in the buffers");
    }
    if (tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor is empty for inserting buffer.");
    }
    buffers_.insert({buffer_idx, tensor});
    return base::error::Success();
}

tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) {
    CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
    return buffers_.at(buffer_idx);
}

const tensor::Tensor& Model::get_buffer(ModelBufferType buffer_idx) const {
    CHECK_GT(buffers_.count(buffer_idx), 0);
    return buffers_.at(buffer_idx);
}

base::Status Model::read_model_file() {
    using namespace base;
    if (model_path_.empty()) {
        return error::PathNotValid("Failed to open the weight file, the model path is empty!");
    }
    int32_t fd = open(model_path_.data(), O_RDONLY);
    if (fd == -1) {
        return error::PathNotValid("Failed to open the weight file " + model_path_ +
                                   " may be the path does not exist!");
    }

    std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(model_path_.data(), "rb"),
                                                       &std::fclose);
    if (!file) {
        return error::PathNotValid("Failed to open the file. The path may be invalid.");
    }

    auto config = ModelConfig{};
    if (fread(&config, sizeof(ModelConfig), 1, file.get()) != 1) {
        return error::ModelParseError(
            "Failed to retrieve the configuration information from the model "
            "file.");
    }
    int32_t immediate_dim = 0;
    if (base::UsesQwen3Layout(model_type_)) {
        if (fread(&immediate_dim, sizeof(int32_t), 1, file.get()) != 1) {
            return error::ModelParseError(
                "Failed to retrieve the qwen3 immediate size information from "
                "the model file.");
        }
    }
    if (is_quant_model_) {
        if (fread(&group_size_, sizeof(int32_t), 1, file.get()) != 1) {
            return error::ModelParseError(
                "Failed to retrieve the group size information from the model "
                "file.");
        }
    }

    auto gen_status = generate_model_infos(config, immediate_dim);
    if (!gen_status) {
        return gen_status;
    }

    if (!is_quant_model_) {
        raw_model_data_ = std::make_shared<RawModelDataFp32>();
    } else {
        raw_model_data_ = std::make_shared<RawModelDataInt8>();
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        return error::ModelParseError(
            "Failed to retrieve the file size information from the model "
            "file.");
    }
    raw_model_data_->file_size = sb.st_size;

    raw_model_data_->fd = fd;
    raw_model_data_->data =
        mmap(nullptr, raw_model_data_->file_size, PROT_READ, MAP_PRIVATE, raw_model_data_->fd, 0);

    if (raw_model_data_->data == MAP_FAILED || raw_model_data_->data == nullptr) {
        return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                      " into memory.");
    }
    size_t model_header_bytes = sizeof(ModelConfig);
    if (base::UsesQwen3Layout(model_type_)) {
        model_header_bytes += sizeof(int32_t);
    }
    if (is_quant_model_) {
        model_header_bytes += sizeof(group_size_);
    }
    raw_model_data_->weight_data = static_cast<int8_t*>(raw_model_data_->data) + model_header_bytes;
    if (raw_model_data_->weight_data == nullptr) {
        LOG(ERROR);
        return error::ModelParseError("Failed to map the weight file " + model_path_ +
                                      " into memory, the pointer to weight start address is null");
    }
    return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config, int32_t immediate_dim) const {
    config_->dim_ = config.dim;
    config_->hidden_dim_ = config.hidden_dim;
    config_->layer_num_ = config.layer_num;
    config_->head_num_ = config.head_num;
    config_->kv_head_num_ = config.kv_head_num;
    config_->seq_len_ = config.seq_len;

    config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
    config_->kv_mul_ = config.head_num / config.kv_head_num;
    config_->head_size_ = config.dim / config.head_num;
    config_->immediate_dim_ = base::UsesQwen3Layout(model_type_) ? immediate_dim : 0;
    if (config.vocab_size > 0) {
        config_->is_shared_weight_ = true;
    } else {
        config_->is_shared_weight_ = false;
    }

    // Qwen tokenizer size and embedding size is mismatched
    // refer: https://github.com/QwenLM/Qwen2.5/issues/29
    // if (std::abs(config.vocab_size) != config_->vocab_size_) {
    //   return base::error::ModelParseError(
    //       "Vocabulary size mismatch between the model file and the token list.");
    // }
    config_->vocab_size_ = std::abs(config.vocab_size);
    return base::error::Success();
}

base::Status Model::create_encode_layer() {
    using namespace base;

    // create token encode decode layer
    if (tokenizer_type_ == TokenizerType::kEncodeSpe) {
        encode_layer_ = std::make_unique<op::SpeEncodeLayer>(this->token_path_, true, false);
    } else if (tokenizer_type_ == TokenizerType::kEncodeBpe) {
        if (use_qwen_tokenizer()) {
            encode_layer_ = std::make_unique<op::QwenEncodeLayer>(this->token_path_, false, false);
        } else {
            encode_layer_ = std::make_unique<op::BpeEncodeLayer>(this->token_path_, true, false);
        }
    }
    if (!encode_layer_) {
        return error::InternalError("Create the encode layer failed.");
    }

    config_->vocab_size_ = encode_layer_->vocab_size();
    if (config_->vocab_size_ <= 0) {
        return error::InternalError("The vocab size param read error from the model file!");
    }
    return error::Success();
}

base::Status Model::gen_model_from_file() {
    using namespace base;
    config_ = std::make_unique<TransformerConfig>();

    // init sentence piece processor
    // google sentence piece
    auto create_encode_status = create_encode_layer();
    if (!create_encode_status) {
        LOG(ERROR) << "Create the encode layer failed!";
        return create_encode_status;
    }
    // mmap
    auto mmap_status = read_model_file();
    if (!mmap_status) {
        LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
        return mmap_status;
    }
    auto layer_create_status = create_layers();
    if (!layer_create_status) {
        LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
        return layer_create_status;
    }

    return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
    CHECK(this->encode_layer_ != nullptr);
    return this->encode_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
    int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
    int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

    float* key_cache_ptr =
        const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
    float* val_cache_ptr =
        const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

    tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                       key_cache_ptr);
    tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr,
                       val_cache_ptr);
    key.set_device_type(device_type_);
    val.set_device_type(device_type_);
    return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
    // pos_tensor：当前位置，通常表示当前 token 的位置
    // embedding_output：embedding 层的输出
    // is_prompt：当前是不是 prompt 阶段
    const int32_t pos = pos_tensor.index<int32_t>(0);
    auto [input_tokens, input_embeddings, input_token_num] = embedding_output;
    UNUSED(input_tokens);
    UNUSED(input_token_num);

    int32_t index = 0;
    if (is_prompt) {
        index = pos;
    }
    const int32_t input_dim = input_width();
    // Prompt stage:
    // input_embeddings shape is [token_num, input_dim], and we slice row `index`.
    // Decode stage:
    // input_embeddings shape is [1, input_dim].
    // In both cases the returned `input` shape is [input_dim].
    std::shared_ptr<base::Buffer> input_emb_buffer = std::make_shared<base::Buffer>(
        input_dim * sizeof(float), nullptr, input_embeddings.ptr<float>(index * input_dim), true);
    tensor::Tensor input(base::DataType::kDataTypeFp32, input_dim);
    input.assign(input_emb_buffer);
    input.set_device_type(device_type_);
    return input;
}

int32_t Model::input_width() const {
    CHECK(config_ != nullptr);
    return config_->dim_;
}

bool Model::use_qwen_tokenizer() const { return false; }
}  // namespace model
