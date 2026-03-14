#include <filesystem>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/llama3.h>

#if defined(LLAMA3_SUPPORT)
constexpr auto kLlamaTokenizerType = base::TokenizerType::kEncodeBpe;
#else
constexpr auto kLlamaTokenizerType = base::TokenizerType::kEncodeSpe;
#endif

TEST(test_llama_model, cpu1) {
#if defined(LLAMA3_SUPPORT)
    const std::filesystem::path checkpoint_path = "/home/fss/llama3/Llama-3.2-1B.bin";
    const std::filesystem::path tokenizer_path = "/home/fss/llama3/tokenizer.json";
#else
    const std::filesystem::path checkpoint_path = "/home/fss/llama2/llama2_7b.bin";
    const std::filesystem::path tokenizer_path = "/home/fss/llama2/tokenizer.model";
#endif
    if (!std::filesystem::exists(checkpoint_path) || !std::filesystem::exists(tokenizer_path)) {
        GTEST_SKIP() << "Local llama test assets are not available.";
    }

    model::LLamaModel model(kLlamaTokenizerType, tokenizer_path.string(), checkpoint_path.string(),
                            false);
    const auto status = model.init(base::DeviceType::kDeviceCPU);
    ASSERT_TRUE(static_cast<bool>(status));

    const auto tokens = model.encode("Hi");
    ASSERT_FALSE(tokens.empty());
}
