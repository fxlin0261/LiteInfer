#include <glog/logging.h>
#include <gtest/gtest.h>
#include <model/llama/llama.h>
#include <filesystem>
#include <memory>
#include <optional>

namespace {
struct LlamaTestAsset {
    std::filesystem::path checkpoint_path;
    std::filesystem::path tokenizer_path;
};

std::optional<LlamaTestAsset> find_llama_test_asset() {
    const LlamaTestAsset llama3_asset{
        "/home/fss/llama3/Llama-3.2-1B.bin",
        "/home/fss/llama3/tokenizer.json",
    };
    if (std::filesystem::exists(llama3_asset.checkpoint_path) &&
        std::filesystem::exists(llama3_asset.tokenizer_path)) {
        return llama3_asset;
    }
    return std::nullopt;
}
}  // namespace

TEST(test_llama_model, cpu1) {
    const auto asset = find_llama_test_asset();
    if (!asset.has_value()) {
        GTEST_SKIP() << "Local llama test assets are not available.";
    }

    auto model = std::make_unique<model::Llama3Model>(asset->tokenizer_path.string(),
                                                      asset->checkpoint_path.string(), false);

    const auto status = model->init(base::DeviceType::kDeviceCPU);
    ASSERT_TRUE(status.ok());
    const auto tokens = model->encode("Hi");
    ASSERT_FALSE(tokens.empty());
}
