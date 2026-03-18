#include "model/model.h"

#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <memory>
#include <utility>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace model {
namespace {
class ScopedFd {
public:
    explicit ScopedFd(int32_t fd = -1) : fd_(fd) {}

    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;

    ScopedFd(ScopedFd&& other) noexcept : fd_(other.release()) {}

    ScopedFd& operator=(ScopedFd&& other) noexcept {
        if (this != &other) {
            reset(other.release());
        }
        return *this;
    }

    ~ScopedFd() { reset(); }

    bool valid() const { return fd_ != -1; }

    int32_t get() const { return fd_; }

    int32_t release() {
        const int32_t fd = fd_;
        fd_ = -1;
        return fd;
    }

    void reset(int32_t fd = -1) {
        if (fd_ != -1) {
            close(fd_);
        }
        fd_ = fd;
    }

private:
    int32_t fd_ = -1;
};

template <typename T>
bool ReadBinaryValue(std::istream& stream, T& value) {
    return static_cast<bool>(stream.read(reinterpret_cast<char*>(&value), sizeof(T)));
}

base::Status ReadModelHeader(std::istream& stream, bool is_quant_model, ModelConfig* config,
                             int32_t* group_size) {
    CHECK(config != nullptr);
    CHECK(group_size != nullptr);

    using namespace base;
    if (!ReadBinaryValue(stream, *config)) {
        return error::ModelParseError(
            "Failed to retrieve the configuration information from the model "
            "file.");
    }
    if (is_quant_model && !ReadBinaryValue(stream, *group_size)) {
        return error::ModelParseError(
            "Failed to retrieve the group size information from the model "
            "file.");
    }
    return error::Success();
}

base::Status ValidateTokenizerVocab(const op::TokenizerLayerBase* tokenizer_layer,
                                    const TransformerConfig& config) {
    using namespace base;
    if (!tokenizer_layer || tokenizer_layer->vocab_size() == config.vocab_size_) {
        return error::Success();
    }

    return error::InvalidArgument("The tokenizer vocab size " +
                                  std::to_string(tokenizer_layer->vocab_size()) +
                                  " does not match the model vocab size " +
                                  std::to_string(config.vocab_size_) + ".");
}

base::Status MapModelData(ScopedFd&& fd, const std::string& model_path, size_t header_bytes,
                          RawModelData* raw_model_data) {
    CHECK(raw_model_data != nullptr);

    using namespace base;
    struct stat sb {};
    if (fstat(fd.get(), &sb) == -1) {
        return error::ModelParseError(
            "Failed to retrieve the file size information from the model "
            "file.");
    }

    raw_model_data->file_size = sb.st_size;
    raw_model_data->fd = fd.release();
    raw_model_data->data =
        mmap(nullptr, raw_model_data->file_size, PROT_READ, MAP_PRIVATE, raw_model_data->fd, 0);
    if (raw_model_data->data == MAP_FAILED || raw_model_data->data == nullptr) {
        return error::ModelParseError("Failed to map the weight file " + model_path +
                                      " into memory.");
    }

    auto* mapped_bytes = static_cast<std::byte*>(raw_model_data->data);
    raw_model_data->weight_data = mapped_bytes + header_bytes;
    return error::Success();
}

std::unique_ptr<op::TokenizerLayerBase> CreateTokenizerLayer(base::TokenizerType tokenizer_type,
                                                             const std::string& token_path) {
    switch (tokenizer_type) {
        case base::TokenizerType::kEncodeSpe:
            return std::make_unique<op::SentencePieceTokenizerLayer>(token_path, true, false);
        case base::TokenizerType::kEncodeBpe:
            return std::make_unique<op::BpeTokenizerLayer>(token_path, true, false);
        default:
            return nullptr;
    }
}

tensor::Tensor MakeExternalFp32TensorView(int32_t element_count, float* data,
                                          base::DeviceType device_type) {
    auto buffer = std::make_shared<base::Buffer>(
        static_cast<size_t>(element_count) * sizeof(float), nullptr, data, true, device_type);
    tensor::Tensor tensor(base::DataType::kDataTypeFp32, element_count);
    CHECK(tensor.assign(buffer));
    return tensor;
}
}  // namespace

Model::Model(base::TokenizerType tokenizer_type, base::ModelType model_type, std::string token_path,
             std::string model_path, bool is_quant_model)
    : tokenizer_type_(tokenizer_type),
      model_type_(model_type),
      token_path_(std::move(token_path)),
      model_path_(std::move(model_path)),
      is_quant_model_(is_quant_model) {}

base::Status Model::set_runtime_max_seq_len(int32_t max_seq_len) {
    if (max_seq_len <= 0) {
        return base::error::InvalidArgument("The runtime max seq_len must be positive.");
    }
    runtime_max_seq_len_ = max_seq_len;
    return base::error::Success();
}

base::Status Model::insert_runtime_tensor(RuntimeTensorType tensor_idx, const tensor::Tensor& tensor) {
    auto& runtime_tensor = runtime_tensors_.at(static_cast<size_t>(tensor_idx));
    if (!runtime_tensor.is_empty()) {
        return base::error::KeyHasExits(std::to_string(int(tensor_idx)) +
                                        " already exists in the runtime tensor storage");
    }
    if (tensor.is_empty()) {
        return base::error::InvalidArgument("The tensor is empty for inserting runtime tensor.");
    }
    runtime_tensor = tensor;
    return base::error::Success();
}

tensor::Tensor& Model::get_runtime_tensor(RuntimeTensorType tensor_idx) {
    auto& runtime_tensor = runtime_tensors_.at(static_cast<size_t>(tensor_idx));
    CHECK(!runtime_tensor.is_empty()) << int(tensor_idx);
    return runtime_tensor;
}

const tensor::Tensor& Model::get_runtime_tensor(RuntimeTensorType tensor_idx) const {
    const auto& runtime_tensor = runtime_tensors_.at(static_cast<size_t>(tensor_idx));
    CHECK(!runtime_tensor.is_empty()) << int(tensor_idx);
    return runtime_tensor;
}

base::Status Model::read_model_file() {
    using namespace base;
    if (model_path_.empty()) {
        return error::PathNotValid("Failed to open the weight file, the model path is empty!");
    }

    ScopedFd fd(open(model_path_.c_str(), O_RDONLY));
    if (!fd.valid()) {
        return error::PathNotValid("Failed to open the weight file " + model_path_ +
                                   " may be the path does not exist!");
    }

    std::ifstream file(model_path_, std::ios::binary);
    if (!file.is_open()) {
        return error::PathNotValid("Failed to open the file. The path may be invalid.");
    }

    ModelConfig config {};
    const Status header_status = ReadModelHeader(file, is_quant_model_, &config, &group_size_);
    if (!header_status.ok()) {
        return header_status;
    }

    const Status model_info_status = generate_model_infos(config);
    if (!model_info_status.ok()) {
        return model_info_status;
    }

    const Status vocab_status = ValidateTokenizerVocab(tokenizer_layer_.get(), *config_);
    if (!vocab_status.ok()) {
        return vocab_status;
    }

    std::shared_ptr<RawModelData> raw_model_data;
    if (is_quant_model_) {
        raw_model_data = std::make_shared<RawModelDataInt8>();
    } else {
        raw_model_data = std::make_shared<RawModelDataFp32>();
    }
    size_t header_bytes = sizeof(ModelConfig);
    if (is_quant_model_) {
        header_bytes += sizeof(int32_t);
    }
    const Status map_status =
        MapModelData(std::move(fd), model_path_, header_bytes, raw_model_data.get());
    if (!map_status.ok()) {
        return map_status;
    }

    raw_model_data_ = std::move(raw_model_data);
    return error::Success();
}

base::Status Model::generate_model_infos(const ModelConfig& config) const {
    CHECK(config_ != nullptr);
    const base::Status validation_status = validate_model_config(config);
    if (!validation_status.ok()) {
        return validation_status;
    }
    auto& model_config = *config_;
    const int32_t runtime_seq_len =
        runtime_max_seq_len_ > 0 && runtime_max_seq_len_ < config.seq_len ? runtime_max_seq_len_
                                                                           : config.seq_len;
    model_config.dim_ = config.dim;
    model_config.hidden_dim_ = config.hidden_dim;
    model_config.layer_num_ = config.layer_num;
    model_config.head_num_ = config.head_num;
    model_config.kv_head_num_ = config.kv_head_num;
    model_config.seq_len_ = runtime_seq_len;
    model_config.kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
    model_config.kv_mul_ = config.head_num / config.kv_head_num;
    model_config.head_size_ = config.dim / config.head_num;
    model_config.is_shared_weight_ = config.vocab_size > 0;
    model_config.vocab_size_ = std::abs(config.vocab_size);
    return base::error::Success();
}

base::Status Model::validate_model_config(const ModelConfig& config) const {
    using namespace base;
    if (config.dim <= 0) {
        return error::InvalidArgument("The model dim must be positive.");
    }
    if (config.hidden_dim <= 0) {
        return error::InvalidArgument("The model hidden_dim must be positive.");
    }
    if (config.layer_num <= 0) {
        return error::InvalidArgument("The model layer_num must be positive.");
    }
    if (config.head_num <= 0) {
        return error::InvalidArgument("The model head_num must be positive.");
    }
    if (config.kv_head_num <= 0) {
        return error::InvalidArgument("The model kv_head_num must be positive.");
    }
    if (config.seq_len <= 0) {
        return error::InvalidArgument("The model seq_len must be positive.");
    }
    if (config.vocab_size == 0) {
        return error::InvalidArgument("The model vocab_size must be non-zero.");
    }
    if (config.dim % config.head_num != 0) {
        return error::InvalidArgument("The model dim must be divisible by head_num.");
    }
    if (config.head_num % config.kv_head_num != 0) {
        return error::InvalidArgument("The model head_num must be divisible by kv_head_num.");
    }
    if ((config.dim * config.kv_head_num) % config.head_num != 0) {
        return error::InvalidArgument("The model kv projection width must resolve to an integer.");
    }
    return error::Success();
}

base::Status Model::create_tokenizer_layer() {
    using namespace base;
    CHECK(config_ != nullptr);

    tokenizer_layer_ = CreateTokenizerLayer(tokenizer_type_, token_path_);
    if (!tokenizer_layer_) {
        return error::InternalError("Create the tokenizer layer failed.");
    }

    const int32_t vocab_size = tokenizer_layer_->vocab_size();
    if (vocab_size <= 0) {
        return error::InternalError("The vocab size param read error from the model file!");
    }
    config_->vocab_size_ = vocab_size;
    return error::Success();
}

base::Status Model::gen_model_from_file() {
    using namespace base;
    config_ = std::make_unique<TransformerConfig>();

    const Status tokenizer_status = create_tokenizer_layer();
    if (!tokenizer_status.ok()) {
        LOG(ERROR) << "Create the tokenizer layer failed!";
        return tokenizer_status;
    }

    const Status read_status = read_model_file();
    if (!read_status.ok()) {
        LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
        return read_status;
    }

    const Status create_layers_status = create_layers();
    if (!create_layers_status.ok()) {
        LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
        return create_layers_status;
    }

    return error::Success();
}

std::vector<int32_t> Model::encode(const std::string& sentence) const {
    CHECK(tokenizer_layer_ != nullptr);
    return tokenizer_layer_->encode(sentence);
}

bool Model::is_sentence_ending(int32_t token_idx) const {
    CHECK(this->tokenizer_layer_ != nullptr);
    return this->tokenizer_layer_->is_sentence_ending(token_idx);
}

std::string Model::decode(int32_t token_idx) const {
    CHECK(this->tokenizer_layer_ != nullptr);
    return this->tokenizer_layer_->decode(token_idx);
}

std::string Model::decode(std::vector<int32_t> token_idxs) const {
    CHECK(this->tokenizer_layer_ != nullptr);
    return this->tokenizer_layer_->decode(token_idxs);
}

std::pair<tensor::Tensor, tensor::Tensor> Model::slice_kv_cache(int32_t layer_idx,
                                                                int32_t token_pos) const {
    // Return tensor views that point at the current token position in the KV cache.
    CHECK(config_ != nullptr);
    const int32_t cache_offset =
        layer_idx * config_->seq_len_ * config_->kv_dim_ + token_pos * config_->kv_dim_;
    float* key_cache_ptr =
        const_cast<float*>(get_runtime_tensor(RuntimeTensorType::kKeyCache).ptr<float>(cache_offset));
    float* val_cache_ptr =
        const_cast<float*>(get_runtime_tensor(RuntimeTensorType::kValueCache).ptr<float>(cache_offset));
    tensor::Tensor key = MakeExternalFp32TensorView(config_->kv_dim_, key_cache_ptr, device_type_);
    tensor::Tensor val = MakeExternalFp32TensorView(config_->kv_dim_, val_cache_ptr, device_type_);
    return {key, val};
}

tensor::Tensor Model::fill_input(const tensor::Tensor& pos_tensor,
                                 const op::EmbeddingOutput& embedding_output,
                                 bool is_prompt) const {
    const int32_t pos = pos_tensor.index<int32_t>(0);
    UNUSED(embedding_output.input_tokens);
    UNUSED(embedding_output.input_token_num);
    // 是 prompt 阶段就取第 pos 行，否则就取第 0 行
    const int32_t input_row = is_prompt ? pos : 0;
    const int32_t input_dim = input_width();
    float* input_ptr = const_cast<float*>(
        embedding_output.input_embeddings.ptr<float>(input_row * input_dim));
    return MakeExternalFp32TensorView(input_dim, input_ptr, device_type_);
}
}  // namespace model
