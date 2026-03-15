#include <filesystem>
#include <optional>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/llama3.h>

namespace {
struct LlamaTestAsset {
    std::filesystem::path checkpoint_path;
    std::filesystem::path tokenizer_path;
    base::TokenizerType tokenizer_type = base::TokenizerType::kEncodeUnknown;
};

std::optional<LlamaTestAsset> find_llama_test_asset() {
    const LlamaTestAsset llama3_asset{
        "/home/fss/llama3/Llama-3.2-1B.bin",
        "/home/fss/llama3/tokenizer.json",
        base::TokenizerType::kEncodeBpe,
    };
    if (std::filesystem::exists(llama3_asset.checkpoint_path) &&
        std::filesystem::exists(llama3_asset.tokenizer_path)) {
        return llama3_asset;
    }

    const LlamaTestAsset llama2_asset{
        "/home/fss/llama2/llama2_7b.bin",
        "/home/fss/llama2/tokenizer.model",
        base::TokenizerType::kEncodeSpe,
    };
    if (std::filesystem::exists(llama2_asset.checkpoint_path) &&
        std::filesystem::exists(llama2_asset.tokenizer_path)) {
        return llama2_asset;
    }
    return std::nullopt;
}
}  // namespace

TEST(test_llama_model, cpu1) {
    const auto asset = find_llama_test_asset();
    if (!asset.has_value()) {
        GTEST_SKIP() << "Local llama test assets are not available.";
    }

    model::LLamaModel model(asset->tokenizer_type, asset->tokenizer_path.string(),
                            asset->checkpoint_path.string(), false);
    const auto status = model.init(base::DeviceType::kDeviceCPU);
    ASSERT_TRUE(static_cast<bool>(status));

    const auto tokens = model.encode("Hi");
    ASSERT_FALSE(tokens.empty());
}
