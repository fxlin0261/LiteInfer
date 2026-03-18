#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "base/alloc.h"
#include "model/llama/llama_model_utils.h"
#include "model/model.h"

namespace {
class FakeEncodeLayer final : public op::TokenizerLayerBase {
public:
    FakeEncodeLayer() : TokenizerLayerBase("fake.model", true, false) {}
    std::vector<int32_t> encode(const std::string& sentence) const override {
        return {101, static_cast<int32_t>(sentence.size())};
    }
    std::string decode(int32_t token_id) const override {
        return "tok_" + std::to_string(token_id);
    }
    std::string decode(const std::vector<int32_t>& token_ids) const override {
        std::ostringstream oss;
        for (size_t i = 0; i < token_ids.size(); ++i) {
            if (i != 0) {
                oss << ",";
            }
            oss << token_ids[i];
        }
        return oss.str();
    }
    bool is_sentence_ending(int32_t token_id) const override { return token_id == 999; }
    int32_t vocab_size() const override { return 2048; }
};

class FakeModel final : public model::Model {
public:
    explicit FakeModel(base::ModelType model_type = base::ModelType::kModelTypeLlama2,
                       base::TokenizerType tokenizer_type = base::TokenizerType::kEncodeSpe,
                       std::string token_path = "token.model",
                       std::string model_path = "model.bin", bool is_quant_model = false)
        : Model(tokenizer_type, model_type, std::move(token_path), std::move(model_path),
                is_quant_model) {
        config_ = std::make_unique<model::TransformerConfig>();
    }
    base::Status init(base::DeviceType device_type, int32_t runtime_max_seq_len = 0) override {
        device_type_ = device_type;
        if (runtime_max_seq_len != 0) {
            return set_runtime_max_seq_len(runtime_max_seq_len);
        }
        return base::error::Success();
    }
    base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         bool is_prompt, int& next) const override {
        UNUSED(input);
        UNUSED(pos_tensor);
        UNUSED(is_prompt);
        next = 0;
        return base::error::Success();
    }

    base::Status forward(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         int& next) const override {
        UNUSED(input);
        UNUSED(pos_tensor);
        next = 0;
        return base::error::Success();
    }

    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const override {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor input_tokens(base::DataType::kDataTypeInt32,
                                    static_cast<int32_t>(tokens.size()), true, alloc);
        tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32,
                                        static_cast<int32_t>(tokens.size()), 1, true, alloc);
        for (size_t i = 0; i < tokens.size(); ++i) {
            input_tokens.index<int32_t>(static_cast<int64_t>(i)) = tokens[i];
        }
        return op::EmbeddingOutput(input_tokens, input_embeddings,
                                   static_cast<int32_t>(tokens.size()));
    }
    void set_tokenizer_layer_for_test(std::unique_ptr<op::TokenizerLayerBase> layer) {
        tokenizer_layer_ = std::move(layer);
    }
    base::Status insert_runtime_tensor_for_test(model::RuntimeTensorType tensor_idx,
                                        const tensor::Tensor& tensor) {
        return insert_runtime_tensor(tensor_idx, tensor);
    }
    base::Status generate_model_infos_for_test(const model::ModelConfig& config) {
        return generate_model_infos(config);
    }
    base::Status read_model_file_for_test() { return read_model_file(); }
    model::TransformerConfig* mutable_config() { return config_.get(); }

private:
    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override {
        UNUSED(pos);
        UNUSED(is_prompt);
        return 0;
    }
    void init_mem() override {}
    base::Status create_layers() override { return base::error::Success(); }
    base::Status create_param_layers() override { return base::error::Success(); }
    base::Status create_nonparam_layers() override { return base::error::Success(); }
    base::Status create_param_quant_layers() override { return base::error::Success(); }
};

tensor::Tensor make_cpu_tensor(base::DataType data_type, const std::vector<int32_t>& dims,
                               void* ptr) {
    return tensor::Tensor::make_external(data_type, dims, ptr, base::DeviceType::kDeviceCPU);
}

std::filesystem::path write_model_file(const model::ModelConfig& config) {
    const auto unique_id =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("kuiper-" + std::to_string(unique_id) + ".bin");
    std::ofstream out(path, std::ios::binary);
    EXPECT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(&config), sizeof(config));
    EXPECT_TRUE(out.good());
    out.close();
    return path;
}
}  // namespace

TEST(test_model_core, generate_model_infos_populates_derived_fields) {
    FakeModel model;
    model::ModelConfig config{};
    config.dim = 12;
    config.hidden_dim = 48;
    config.layer_num = 2;
    config.head_num = 6;
    config.kv_head_num = 3;
    config.vocab_size = -32000;
    config.seq_len = 16;
    ASSERT_TRUE(model.generate_model_infos_for_test(config).ok());
    const auto* transformer = model.mutable_config();
    ASSERT_NE(transformer, nullptr);
    EXPECT_EQ(transformer->dim_, 12);
    EXPECT_EQ(transformer->hidden_dim_, 48);
    EXPECT_EQ(transformer->layer_num_, 2);
    EXPECT_EQ(transformer->head_num_, 6);
    EXPECT_EQ(transformer->kv_head_num_, 3);
    EXPECT_EQ(transformer->seq_len_, 16);
    EXPECT_EQ(transformer->kv_dim_, 6);
    EXPECT_EQ(transformer->kv_mul_, 2);
    EXPECT_EQ(transformer->head_size_, 2);
    EXPECT_EQ(transformer->vocab_size_, 32000);
    EXPECT_FALSE(transformer->is_shared_weight_);
    config.vocab_size = 32000;
    ASSERT_TRUE(model.generate_model_infos_for_test(config).ok());
    EXPECT_TRUE(model.mutable_config()->is_shared_weight_);
}

TEST(test_model_core, runtime_max_seq_len_caps_model_seq_len_without_expanding_it) {
    FakeModel model;
    model::ModelConfig config{};
    config.dim = 12;
    config.hidden_dim = 48;
    config.layer_num = 2;
    config.head_num = 6;
    config.kv_head_num = 3;
    config.vocab_size = -32000;
    config.seq_len = 131072;

    ASSERT_TRUE(model.set_runtime_max_seq_len(8192).ok());
    ASSERT_TRUE(model.generate_model_infos_for_test(config).ok());
    EXPECT_EQ(model.mutable_config()->seq_len_, 8192);

    ASSERT_TRUE(model.set_runtime_max_seq_len(262144).ok());
    ASSERT_TRUE(model.generate_model_infos_for_test(config).ok());
    EXPECT_EQ(model.mutable_config()->seq_len_, 131072);
}

TEST(test_model_core, init_can_apply_runtime_max_seq_len) {
    FakeModel model;
    model::ModelConfig config{};
    config.dim = 12;
    config.hidden_dim = 48;
    config.layer_num = 2;
    config.head_num = 6;
    config.kv_head_num = 3;
    config.vocab_size = -32000;
    config.seq_len = 131072;

    ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU, 8192).ok());
    ASSERT_TRUE(model.generate_model_infos_for_test(config).ok());
    EXPECT_EQ(model.mutable_config()->seq_len_, 8192);
}

TEST(test_model_core, runtime_max_seq_len_requires_positive_value) {
    FakeModel model;
    const auto status = model.set_runtime_max_seq_len(0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), base::StatusCode::kInvalidArgument);
}

TEST(test_model_core, generate_model_infos_rejects_invalid_model_config) {
    auto expect_invalid = [](const model::ModelConfig& config, const char* message_fragment) {
        FakeModel model;
        const auto status = model.generate_model_infos_for_test(config);
        EXPECT_FALSE(status.ok());
        EXPECT_EQ(status.code(), base::StatusCode::kInvalidArgument);
        EXPECT_NE(std::string(status.message()).find(message_fragment), std::string::npos);
    };

    model::ModelConfig zero_heads{};
    zero_heads.dim = 16;
    zero_heads.hidden_dim = 32;
    zero_heads.layer_num = 2;
    zero_heads.head_num = 0;
    zero_heads.kv_head_num = 2;
    zero_heads.vocab_size = 32000;
    zero_heads.seq_len = 8;
    expect_invalid(zero_heads, "head_num must be positive");

    model::ModelConfig bad_head_split = zero_heads;
    bad_head_split.dim = 18;
    bad_head_split.head_num = 6;
    bad_head_split.kv_head_num = 4;
    expect_invalid(bad_head_split, "head_num must be divisible by kv_head_num");

    model::ModelConfig bad_dim_split = zero_heads;
    bad_dim_split.head_num = 6;
    bad_dim_split.kv_head_num = 3;
    bad_dim_split.dim = 14;
    expect_invalid(bad_dim_split, "dim must be divisible by head_num");

    model::ModelConfig zero_vocab = zero_heads;
    zero_vocab.head_num = 4;
    zero_vocab.kv_head_num = 2;
    zero_vocab.vocab_size = 0;
    expect_invalid(zero_vocab, "vocab_size must be non-zero");
}

TEST(test_model_core, insert_runtime_tensor_rejects_empty_tensor_and_duplicates) {
    FakeModel model;
    tensor::Tensor empty;
    auto status = model.insert_runtime_tensor_for_test(model::RuntimeTensorType::kInputTokens, empty);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), base::StatusCode::kInvalidArgument);
    std::vector<int32_t> token_data{1, 2, 3};
    auto tensor = make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_data.data());
    ASSERT_TRUE(model.insert_runtime_tensor_for_test(model::RuntimeTensorType::kInputTokens, tensor).ok());
    status = model.insert_runtime_tensor_for_test(model::RuntimeTensorType::kInputTokens, tensor);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), base::StatusCode::kAlreadyExists);
}

TEST(test_model_core, encode_decode_and_sentence_ending_delegate_to_encode_layer) {
    FakeModel model;
    model.set_tokenizer_layer_for_test(std::make_unique<FakeEncodeLayer>());
    const auto encoded = model.encode("hello");
    ASSERT_EQ(encoded.size(), 2U);
    EXPECT_EQ(encoded[0], 101);
    EXPECT_EQ(encoded[1], 5);
    EXPECT_EQ(model.decode(7), "tok_7");
    EXPECT_EQ((model.decode(std::vector<int32_t>{3, 4, 5})), "3,4,5");
    EXPECT_TRUE(model.is_sentence_ending(999));
    EXPECT_FALSE(model.is_sentence_ending(7));
}

TEST(test_model_core, fill_input_uses_prompt_position_or_first_generation_token) {
    auto run_case = [](base::ModelType model_type, base::TokenizerType tokenizer_type) {
        FakeModel model(model_type, tokenizer_type);
        ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU).ok());
        constexpr int32_t embed_dim = 4;
        model.mutable_config()->dim_ = embed_dim;
        std::vector<int32_t> token_ids{11, 22, 33};
        std::vector<float> embedding_data{
            1.f,   2.f,   3.f,   4.f,   //
            10.f,  20.f,  30.f,  40.f,  //
            100.f, 200.f, 300.f, 400.f,
        };

        auto input_tokens = make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_ids.data());
        auto input_embeddings =
            make_cpu_tensor(base::DataType::kDataTypeFp32, {3, embed_dim}, embedding_data.data());
        op::EmbeddingOutput embedding_output(input_tokens, input_embeddings,
                                            static_cast<int32_t>(token_ids.size()));
        int32_t pos_value = 2;
        auto pos_tensor = make_cpu_tensor(base::DataType::kDataTypeInt32, {1}, &pos_value);
        tensor::Tensor prompt_input = model.fill_input(pos_tensor, embedding_output, true);
        ASSERT_EQ(prompt_input.size(), static_cast<size_t>(embed_dim));
        EXPECT_EQ(prompt_input.ptr<float>(), embedding_data.data() + 2 * embed_dim);
        EXPECT_FLOAT_EQ(prompt_input.index<float>(0), 100.f);
        EXPECT_FLOAT_EQ(prompt_input.index<float>(3), 400.f);
        tensor::Tensor generation_input = model.fill_input(pos_tensor, embedding_output, false);
        ASSERT_EQ(generation_input.size(), static_cast<size_t>(embed_dim));
        EXPECT_EQ(generation_input.ptr<float>(), embedding_data.data());
        EXPECT_FLOAT_EQ(generation_input.index<float>(0), 1.f);
        EXPECT_FLOAT_EQ(generation_input.index<float>(3), 4.f);
    };

    run_case(base::ModelType::kModelTypeLlama2, base::TokenizerType::kEncodeSpe);
    run_case(base::ModelType::kModelTypeLlama3, base::TokenizerType::kEncodeBpe);
}

TEST(test_model_core, slice_kv_cache_returns_tensor_views_into_backing_storage) {
    FakeModel model;
    ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU).ok());
    model.mutable_config()->seq_len_ = 4;
    model.mutable_config()->kv_dim_ = 3;
    std::vector<float> key_cache(24);
    std::iota(key_cache.begin(), key_cache.end(), 0.f);
    std::vector<float> value_cache(24);
    std::iota(value_cache.begin(), value_cache.end(), 100.f);
    auto key_tensor = make_cpu_tensor(base::DataType::kDataTypeFp32, {24}, key_cache.data());
    auto value_tensor = make_cpu_tensor(base::DataType::kDataTypeFp32, {24}, value_cache.data());
    ASSERT_TRUE(model.insert_runtime_tensor_for_test(model::RuntimeTensorType::kKeyCache, key_tensor).ok());
    ASSERT_TRUE(
        model.insert_runtime_tensor_for_test(model::RuntimeTensorType::kValueCache, value_tensor).ok());
    auto [key_view, value_view] = model.slice_kv_cache(1, 2);
    ASSERT_EQ(key_view.size(), 3U);
    ASSERT_EQ(value_view.size(), 3U);
    EXPECT_EQ(key_view.ptr<float>(), key_cache.data() + 18);
    EXPECT_EQ(value_view.ptr<float>(), value_cache.data() + 18);
    EXPECT_FLOAT_EQ(key_view.index<float>(0), 18.f);
    EXPECT_FLOAT_EQ(value_view.index<float>(2), 120.f);
    key_view.index<float>(1) = -7.f;
    value_view.index<float>(0) = -3.f;
    EXPECT_FLOAT_EQ(key_cache[19], -7.f);
    EXPECT_FLOAT_EQ(value_cache[18], -3.f);
}

TEST(test_model_core, read_model_file_rejects_tokenizer_and_model_vocab_mismatch) {
    model::ModelConfig config{};
    config.dim = 16;
    config.hidden_dim = 32;
    config.layer_num = 2;
    config.head_num = 4;
    config.kv_head_num = 4;
    config.vocab_size = 4096;
    config.seq_len = 8;

    const auto model_path = write_model_file(config);
    {
        FakeModel model(base::ModelType::kModelTypeLlama2, base::TokenizerType::kEncodeSpe,
                        "token.model", model_path.string());
        model.set_tokenizer_layer_for_test(std::make_unique<FakeEncodeLayer>());
        const auto status = model.read_model_file_for_test();
        EXPECT_FALSE(status.ok());
        EXPECT_EQ(status.code(), base::StatusCode::kInvalidArgument);
        EXPECT_NE(std::string(status.message()).find("tokenizer vocab size"), std::string::npos);
    }
    std::filesystem::remove(model_path);
}

TEST(test_model_core, legacy_quantized_weights_layout_skips_classifier_blob_when_shared) {
    std::vector<int8_t> weights(256, 0);
    model::RawModelDataInt8 raw_model_data;
    raw_model_data.weight_data = weights.data();

    const auto shared_layout =
        model::detail::ResolveLegacyQuantizedWeightsLayout(raw_model_data, 16, 8, 4, 4, true);
    EXPECT_EQ(shared_layout.classifier_weight, weights.data() + 16);
    EXPECT_EQ(shared_layout.embedding_weight, weights.data() + 16);
    EXPECT_FALSE(shared_layout.classifier_is_quantized);

    const auto unshared_layout =
        model::detail::ResolveLegacyQuantizedWeightsLayout(raw_model_data, 16, 8, 4, 4, false);
    const size_t classifier_bytes = model::detail::LegacyQuantizedTensorBytes(8, 4, 4);
    EXPECT_EQ(unshared_layout.classifier_weight, weights.data() + 16);
    EXPECT_EQ(unshared_layout.embedding_weight, weights.data() + 16 + classifier_bytes);
    EXPECT_TRUE(unshared_layout.classifier_is_quantized);
}
