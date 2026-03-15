#ifndef KUIPER_INCLUDE_OP_TOKENIZER_LAYER_H_
#define KUIPER_INCLUDE_OP_TOKENIZER_LAYER_H_
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <absl/strings/str_split.h>
#include <sentencepiece_processor.h>
#include "tokenizer/tiktoken.h"
#include "base/unordered_dense.h"
#include "layer.h"
#include "nlohmann/json.hpp"
namespace op {

class TokenizerLayerBase : public Layer {
public:
    explicit TokenizerLayerBase(std::string token_model_path, bool has_bos, bool has_eos)
        : Layer(base::DeviceType::kDeviceCPU, LayerType::kLayerEncode, "Encode"),
          has_bos_(has_bos),
          has_eos_(has_eos),
          token_model_path_(std::move(token_model_path)) {}

    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

    virtual std::string decode(int32_t token_id) const = 0;

    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;

    virtual bool is_sentence_ending(int32_t token_id) const = 0;

    virtual int32_t vocab_size() const = 0;

protected:
    bool has_bos_ = true;
    bool has_eos_ = false;
    std::string token_model_path_;
};

class SentencePieceTokenizerLayer : public TokenizerLayerBase {
public:
    explicit SentencePieceTokenizerLayer(std::string token_model_path, bool has_bos, bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> spe;
};

class BpeTokenizerLayer : public TokenizerLayerBase {
public:
    explicit BpeTokenizerLayer(std::string token_model_path, bool has_bos, bool has_eos);

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_id) const override;

    std::string decode(const std::vector<int32_t>& token_ids) const override;

    bool is_sentence_ending(int32_t token_id) const override;

    int32_t vocab_size() const override;

protected:
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t stop_token1_ = -1;
    int32_t stop_token2_ = -1;
    int32_t num_token_ = 0;
    std::unique_ptr<tiktoken::tiktoken> tiktoken_;
};

class QwenTokenizerLayer : public BpeTokenizerLayer {
public:
    explicit QwenTokenizerLayer(std::string token_model_path, bool has_bos, bool has_eos);
};

}  // namespace op
#endif  // KUIPER_INCLUDE_OP_TOKENIZER_LAYER_H_
