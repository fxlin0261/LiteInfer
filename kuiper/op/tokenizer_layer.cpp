#include "op/tokenizer_layer.h"
#include <glog/logging.h>
#include "base/unicode.h"
namespace op {

std::string SentencePieceTokenizerLayer::decode(int32_t token_id) const {
    CHECK(spe != nullptr);
    std::vector<int32_t> token_ids{token_id};
    return this->spe->DecodeIds(token_ids);
}

std::string SentencePieceTokenizerLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(spe != nullptr);
    return this->spe->DecodeIds(token_ids);
}

SentencePieceTokenizerLayer::SentencePieceTokenizerLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : TokenizerLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using namespace sentencepiece::util;
    spe = std::make_unique<sentencepiece::SentencePieceProcessor>();
    auto rc = spe->Load(token_model_path_);
    if (rc.code() != StatusCode::kOk) {
        LOG(FATAL)
            << "The token model path is not valid, please check the path and type of token model.";
    }
}

std::vector<int32_t> SentencePieceTokenizerLayer::encode(const std::string& sentence) const {
    CHECK(spe != nullptr);
    // SentencePiece 编码，可按需补 BOS/EOS。
    std::vector<int32_t> input_ids = spe->EncodeAsIds(sentence);
    if (has_bos_) {
        input_ids.insert(input_ids.begin(), spe->bos_id());
    }
    if (has_eos_) {
        input_ids.push_back(spe->eos_id());
    }
    return input_ids;
}

bool SentencePieceTokenizerLayer::is_sentence_ending(int32_t token_id) const {
    CHECK(this->spe != nullptr);
    return token_id == this->spe->eos_id();
}

int32_t SentencePieceTokenizerLayer::vocab_size() const {
    CHECK(spe != nullptr);
    return spe->GetPieceSize();
}

static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeTokenizerLayer::BpeTokenizerLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : TokenizerLayerBase(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path_);
    CHECK(f.is_open())
        << "The token model path is not valid, please check the path and type of token model.";
    json data;
    try {
        data = json::parse(f);
    } catch (json::parse_error&) {
        LOG(FATAL)
            << "The token model path is not valid, please check the path and type of token model.";
    }

    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for (const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();
    for (const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    bos_id_ = special_tokens["<|begin_of_text|>"];
    eos_id_ = special_tokens["<|end_of_text|>"];
    stop_token1_ = eos_id_;
    stop_token2_ = special_tokens["<|eot_id|>"];

    num_token_ = encoder.size() + special_tokens.size();
    tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeTokenizerLayer::encode(const std::string& sentence) const {
    CHECK(this->tiktoken_ != nullptr);
    std::map<std::string, std::string> replacements;
    replacements[" "] = "Ġ";
    std::string s = absl::StrReplaceAll(sentence, replacements);
    auto input_ids = this->tiktoken_->encode(s);

    if (has_bos_) {
        input_ids.insert(input_ids.begin(), bos_id_);
    }
    if (has_eos_) {
        input_ids.push_back(eos_id_);
    }
    return input_ids;
}

std::string BpeTokenizerLayer::decode(int32_t token_id) const { return ""; }

std::string BpeTokenizerLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(this->tiktoken_ != nullptr);
    auto s = tiktoken_->decode(token_ids);
    std::map<std::string, std::string> reverse_replacements;
    reverse_replacements["Ġ"] = " ";
    const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
    return sentence;
}

bool BpeTokenizerLayer::is_sentence_ending(int32_t token_id) const {
    if (token_id == stop_token1_ || token_id == stop_token2_) {
        return true;
    } else {
        return false;
    }
}

int32_t BpeTokenizerLayer::vocab_size() const {
    CHECK(this->tiktoken_ != nullptr);
    return num_token_;
}

QwenTokenizerLayer::QwenTokenizerLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : BpeTokenizerLayer(std::move(token_model_path), has_bos, has_eos) {
    using json = nlohmann::json;
    std::ifstream f(token_model_path_);

    json data = json::parse(f);
    const auto& datas = data["added_tokens"];
    ankerl::unordered_dense::map<std::string, int> special_tokens;
    for (const auto& data1 : datas) {
        int id = data1["id"];
        std::string content = data1["content"];
        special_tokens.insert({content, id});
    }

    ankerl::unordered_dense::map<std::string, int> encoder;
    const auto& vocabs = data["model"]["vocab"];
    const auto& vocab_items = vocabs.items();
    for (const auto& v : vocab_items) {
        const auto cpts = unicode_cpts_from_utf8(v.key());
        std::string key;
        for (const auto cpt : cpts) {
            const auto utf8 = unicode_cpt_to_utf8(cpt);
            key += unicode_utf8_to_byte(utf8);
        }
        const int32_t id = v.value();
        encoder[key] = id;
    }
    bos_id_ = special_tokens["<|im_start|>"];
    eos_id_ = special_tokens["<|im_end|>"];
    stop_token1_ = eos_id_;
    stop_token2_ = special_tokens["<|endoftext|>"];

    num_token_ = encoder.size() + special_tokens.size();
    tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}
}  // namespace op
