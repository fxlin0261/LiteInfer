#include <filesystem>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/llama3.h>

TEST(test_llama_model, cpu1) {
    const std::filesystem::path checkpoint_path = "/home/fss/llama2/llama2_7b.bin";
    const std::filesystem::path tokenizer_path = "/home/fss/llama2/tokenizer.model";
    if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(tokenizer_path)) {
        GTEST_SKIP() << "Local llama2 test assets are not available.";
    }

    model::LLama2Model model(base::TokenizerType::kEncodeSpe, tokenizer_path.string(),
                             checkpoint_path.string(), false);
    const auto status = model.init(base::DeviceType::kDeviceCPU);
    ASSERT_TRUE(static_cast<bool>(status));

    const auto tokens = model.encode("Hi");
    ASSERT_FALSE(tokens.empty());
}
