#include <gtest/gtest.h>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include "base/alloc.h"
#include "model/model.h"

namespace {

class FakeEncodeLayer final : public op::EncodeLayerBase {
public:
    FakeEncodeLayer() : EncodeLayerBase("fake.model", true, false) {}

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
    FakeModel()
        : Model(base::TokenizerType::kEncodeSpe, base::ModelType::kModelTypeLLama2, "token.model",
                "model.bin", false) {
        config_ = std::make_unique<model::TransformerConfig>();
    }

    base::Status init(base::DeviceType device_type) override {
        device_type_ = device_type;
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
        tensor::Tensor input_token_num(base::DataType::kDataTypeInt32,
                                       static_cast<int32_t>(tokens.size()), true, alloc);
        for (size_t i = 0; i < tokens.size(); ++i) {
            input_tokens.index<int32_t>(static_cast<int64_t>(i)) = tokens[i];
        }
        return op::EmbeddingOutput(input_tokens, input_embeddings, input_token_num);
    }

    void set_encode_layer_for_test(std::unique_ptr<op::EncodeLayerBase> layer) {
        encode_layer_ = std::move(layer);
    }

    base::Status insert_buffer_for_test(model::ModelBufferType buffer_idx,
                                        const tensor::Tensor& tensor) {
        return insert_buffer(buffer_idx, tensor);
    }

    base::Status generate_model_infos_for_test(const model::ModelConfig& config) {
        return generate_model_infos(config);
    }

    model::TransformerConfig* mutable_config() { return config_.get(); }

private:
    int32_t post_processing(const tensor::Tensor& pos, bool is_prompt) const override {
        UNUSED(pos);
        UNUSED(is_prompt);
        return 0;
    }

    void init_mem() override {}

    base::Status create_layers() override { return base::error::Success(); }

    void create_param_layers() override {}

    void create_nonparam_layers() override {}

    void create_param_quant_layers() override {}
};

tensor::Tensor make_cpu_tensor(base::DataType data_type, const std::vector<int32_t>& dims,
                               void* ptr) {
    tensor::Tensor tensor(data_type, dims, false, nullptr, ptr);
    tensor.set_device_type(base::DeviceType::kDeviceCPU);
    return tensor;
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

    ASSERT_TRUE(model.generate_model_infos_for_test(config));
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
    ASSERT_TRUE(model.generate_model_infos_for_test(config));
    EXPECT_TRUE(model.mutable_config()->is_shared_weight_);
}

TEST(test_model_core, insert_buffer_rejects_empty_tensor_and_duplicates) {
    FakeModel model;

    tensor::Tensor empty;
    auto status = model.insert_buffer_for_test(model::ModelBufferType::kInputTokens, empty);
    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), base::StatusCode::kInvalidArgument);

    std::vector<int32_t> token_data{1, 2, 3};
    auto tensor = make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_data.data());
    ASSERT_TRUE(model.insert_buffer_for_test(model::ModelBufferType::kInputTokens, tensor));

    status = model.insert_buffer_for_test(model::ModelBufferType::kInputTokens, tensor);
    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), base::StatusCode::kKeyValueHasExist);
}

TEST(test_model_core, encode_decode_and_sentence_ending_delegate_to_encode_layer) {
    FakeModel model;
    model.set_encode_layer_for_test(std::make_unique<FakeEncodeLayer>());

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
    FakeModel model;
    ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU));

    constexpr int32_t embed_dim = 4;
#if defined(QWEN3_SUPPORT)
    model.mutable_config()->hidden_dim_ = embed_dim;
#else
    model.mutable_config()->dim_ = embed_dim;
#endif

    std::vector<int32_t> token_ids{11, 22, 33};
    std::vector<int32_t> token_count_placeholder{0, 0, 0};
    std::vector<float> embedding_data{
        1.f,   2.f,   3.f,   4.f,   //
        10.f,  20.f,  30.f,  40.f,  //
        100.f, 200.f, 300.f, 400.f,
    };

    auto input_tokens = make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_ids.data());
    auto input_token_num =
        make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_count_placeholder.data());
    auto input_embeddings =
        make_cpu_tensor(base::DataType::kDataTypeFp32, {3, embed_dim}, embedding_data.data());
    op::EmbeddingOutput embedding_output(input_tokens, input_embeddings, input_token_num);

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
}

TEST(test_model_core, slice_kv_cache_returns_tensor_views_into_backing_storage) {
    FakeModel model;
    ASSERT_TRUE(model.init(base::DeviceType::kDeviceCPU));
    model.mutable_config()->seq_len_ = 4;
    model.mutable_config()->kv_dim_ = 3;

    std::vector<float> key_cache(24);
    std::iota(key_cache.begin(), key_cache.end(), 0.f);
    std::vector<float> value_cache(24);
    std::iota(value_cache.begin(), value_cache.end(), 100.f);

    auto key_tensor = make_cpu_tensor(base::DataType::kDataTypeFp32, {24}, key_cache.data());
    auto value_tensor = make_cpu_tensor(base::DataType::kDataTypeFp32, {24}, value_cache.data());
    ASSERT_TRUE(model.insert_buffer_for_test(model::ModelBufferType::kKeyCache, key_tensor));
    ASSERT_TRUE(model.insert_buffer_for_test(model::ModelBufferType::kValueCache, value_tensor));

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
