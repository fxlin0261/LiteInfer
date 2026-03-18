#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "base/alloc.h"
#include "model/generation.h"
#include "op/embedding.h"
#include "base/tensor.h"

namespace {
class FakeGenerationModel {
public:
    FakeGenerationModel(std::vector<int32_t> prompt_tokens, std::vector<int32_t> predicted_tokens,
                        int32_t eos_token)
        : prompt_tokens_(std::move(prompt_tokens)),
          predicted_tokens_(std::move(predicted_tokens)),
          eos_token_(eos_token),
          pos_tensor_(base::DataType::kDataTypeInt32, 1, true,
                      base::CPUDeviceAllocatorFactory::get_instance()) {}
    std::vector<int32_t> encode(const std::string& sentence) const {
        UNUSED(sentence);
        return prompt_tokens_;
    }
    const tensor::Tensor& get_runtime_tensor(model::RuntimeTensorType tensor_idx) const {
        CHECK_EQ(tensor_idx, model::RuntimeTensorType::kInputPos);
        return pos_tensor_;
    }
    op::EmbeddingOutput embedding(const std::vector<int>& tokens) const {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor input_tokens(base::DataType::kDataTypeInt32,
                                    static_cast<int32_t>(tokens.size()), true, alloc);
        tensor::Tensor input_embeddings(base::DataType::kDataTypeFp32,
                                        static_cast<int32_t>(tokens.size()), 1, true, alloc);

        for (size_t i = 0; i < tokens.size(); ++i) {
            input_tokens.index<int32_t>(static_cast<int64_t>(i)) = tokens.at(i);
            input_embeddings.index<float>(static_cast<int64_t>(i)) =
                static_cast<float>(tokens.at(i));
        }
        return op::EmbeddingOutput(input_tokens, input_embeddings,
                                   static_cast<int32_t>(tokens.size()));
    }
    tensor::Tensor fill_input(const tensor::Tensor& pos_tensor,
                              const op::EmbeddingOutput& embedding_output,
                              bool is_prompt) const {
        auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
        tensor::Tensor input(base::DataType::kDataTypeFp32, 1, true, alloc);
        const int32_t index = is_prompt ? pos_tensor.index<int32_t>(0) : 0;
        input.index<float>(0) = embedding_output.input_embeddings.index<float>(index);
        return input;
    }
    base::Status predict(const tensor::Tensor& input, const tensor::Tensor& pos_tensor,
                         bool is_prompt, int& next) const {
        seen_inputs_.push_back(static_cast<int32_t>(input.index<float>(0)));
        seen_positions_.push_back(pos_tensor.index<int32_t>(0));
        seen_prompt_flags_.push_back(is_prompt ? 1 : 0);
        if (predict_call_count_ >= predicted_tokens_.size()) {
            return base::error::InvalidArgument("No prepared prediction for this step.");
        }
        next = predicted_tokens_.at(predict_call_count_);
        ++predict_call_count_;
        return base::error::Success();
    }
    bool is_sentence_ending(int32_t token_idx) const { return token_idx == eos_token_; }
    std::string decode(const std::vector<int32_t>& token_idxs) const {
        std::string decoded;
        for (size_t i = 0; i < token_idxs.size(); ++i) {
            if (!decoded.empty()) {
                decoded += " ";
            }
            decoded += std::to_string(token_idxs.at(i));
        }
        return decoded;
    }
    const std::vector<int32_t>& seen_inputs() const { return seen_inputs_; }
    const std::vector<int32_t>& seen_positions() const { return seen_positions_; }
    const std::vector<int32_t>& seen_prompt_flags() const { return seen_prompt_flags_; }

private:
    std::vector<int32_t> prompt_tokens_;
    std::vector<int32_t> predicted_tokens_;
    int32_t eos_token_ = -1;
    mutable size_t predict_call_count_ = 0;
    mutable std::vector<int32_t> seen_inputs_;
    mutable std::vector<int32_t> seen_positions_;
    mutable std::vector<int32_t> seen_prompt_flags_;
    mutable tensor::Tensor pos_tensor_;
};

TEST(test_generation, single_token_prompt_switches_to_decode_path_cleanly) {
    FakeGenerationModel model({42}, {7, 99}, 99);
    app::GenerationState result;

    const auto status =
        app::RunGeneration(model, model.encode("ignored"), 4,
                           app::CollectPromptAndGeneratedTokens, &result);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(result.executed_steps, 1);
    EXPECT_EQ(result.words, std::vector<int32_t>({7}));
    EXPECT_EQ(model.seen_inputs(), std::vector<int32_t>({42, 7}));
    EXPECT_EQ(model.seen_positions(), std::vector<int32_t>({0, 1}));
    EXPECT_EQ(model.seen_prompt_flags(), std::vector<int32_t>({0, 0}));
}

TEST(test_generation, prompt_tokens_are_consumed_before_generated_tokens) {
    FakeGenerationModel model({10, 11, 12}, {50, 51, 52}, -1);
    app::GenerationState result;

    const auto status =
        app::RunGeneration(model, model.encode("ignored"), 3,
                           app::CollectPromptAndGeneratedTokens, &result);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(result.executed_steps, 3);
    EXPECT_EQ(result.words, std::vector<int32_t>({11, 12, 52}));
    EXPECT_EQ(model.seen_inputs(), std::vector<int32_t>({10, 11, 12}));
    EXPECT_EQ(model.seen_positions(), std::vector<int32_t>({0, 1, 2}));
    EXPECT_EQ(model.seen_prompt_flags(), std::vector<int32_t>({1, 1, 0}));
}
}  // namespace
