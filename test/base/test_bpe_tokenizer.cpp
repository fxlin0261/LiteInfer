#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "tokenizer/tokenizer_layer.h"

namespace {
std::filesystem::path WriteMiniTokenizerJson() {
    const std::filesystem::path output =
        std::filesystem::temp_directory_path() / "liteinfer_test_tokenizer.json";
    std::ofstream file(output);
    CHECK(file.is_open());
    file << R"JSON(
{
  "added_tokens": [
    {"id": 10, "content": "<|begin_of_text|>"},
    {"id": 11, "content": "<|end_of_text|>"},
    {"id": 12, "content": "<|eot_id|>"}
  ],
  "model": {
    "vocab": {
      "A": 0,
      "\u0120A": 1,
      "B": 2,
      "\u0120B": 3
    }
  }
}
)JSON";
    return output;
}
}  // namespace

TEST(test_bpe_tokenizer, encode_keeps_raw_spaces_for_byte_level_vocab) {
    const auto tokenizer_path = WriteMiniTokenizerJson();
    op::BpeTokenizerLayer tokenizer(tokenizer_path.string(), false, false);

    EXPECT_EQ((tokenizer.encode("A")), (std::vector<int32_t>{0}));
    EXPECT_EQ((tokenizer.encode(" A")), (std::vector<int32_t>{1}));
    EXPECT_EQ((tokenizer.encode(" B")), (std::vector<int32_t>{3}));
}

TEST(test_bpe_tokenizer, encode_handles_special_tokens_without_rewriting_following_text) {
    const auto tokenizer_path = WriteMiniTokenizerJson();
    op::BpeTokenizerLayer tokenizer(tokenizer_path.string(), false, false);

    EXPECT_EQ((tokenizer.encode("<|begin_of_text|> A<|eot_id|>")),
              (std::vector<int32_t>{10, 1, 12}));
    EXPECT_EQ(tokenizer.decode(std::vector<int32_t>{10, 1, 12}),
              "<|begin_of_text|> A<|eot_id|>");
}
