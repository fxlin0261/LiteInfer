#include "tokenizer/tokenizer_layer.h"
#include <glog/logging.h>
#include "tokenizer/unicode_byte_fallback.h"
#include "base/unicode_utf8.h"

namespace op {
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
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
    auto input_ids = this->tiktoken_->encode(sentence);

    if (has_bos_ && (input_ids.empty() || input_ids.front() != bos_id_)) {
        input_ids.insert(input_ids.begin(), bos_id_);
    }
    if (has_eos_ && (input_ids.empty() || input_ids.back() != eos_id_)) {
        input_ids.push_back(eos_id_);
    }
    return input_ids;
}

std::string BpeTokenizerLayer::decode(int32_t token_id) const {
    return decode(std::vector<int32_t>{token_id});
}

std::string BpeTokenizerLayer::decode(const std::vector<int32_t>& token_ids) const {
    CHECK(this->tiktoken_ != nullptr);
    return tiktoken_->decode(token_ids);
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

int32_t BpeTokenizerLayer::bos_token_id() const { return bos_id_; }
}  // namespace op
