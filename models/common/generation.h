#ifndef KUIPER_MODELS_COMMON_GENERATION_H_
#define KUIPER_MODELS_COMMON_GENERATION_H_

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <glog/logging.h>
#include "model/core/model.h"

namespace app {
struct GenerationState {
    int32_t next = -1;
    bool is_prompt = true;
    std::vector<int32_t> words;
};

struct GenerationResult {
    int32_t steps = 0;
    GenerationState state;
};
inline void CollectPromptAndGeneratedTokens(GenerationState& state,
                                            const std::vector<int32_t>& prompt_tokens, int32_t pos,
                                            int32_t prompt_len) {
    if (pos < prompt_len - 1) {
        state.next = prompt_tokens.at(pos + 1);
    } else {
        state.is_prompt = false;
    }
    state.words.push_back(state.next);
}

template <typename TokenCollector>
void CollectToken(TokenCollector&& collect_token, GenerationState& state,
                  const std::vector<int32_t>& tokens, int32_t pos, int32_t prompt_len) {
    if constexpr (std::is_invocable_v<TokenCollector, GenerationState&,
                                      const std::vector<int32_t>&, int32_t, int32_t>) {
        std::forward<TokenCollector>(collect_token)(state, tokens, pos, prompt_len);
    } else {
        std::forward<TokenCollector>(collect_token)(state, tokens, pos);
    }
}

template <typename ModelT, typename TokenCollector>
GenerationResult RunGeneration(const ModelT& model, std::vector<int32_t> tokens, int total_steps,
                               GenerationState state, TokenCollector&& collect_token) {
    GenerationResult result;
    STATUS_CHECK(RunGeneration(model, std::move(tokens), total_steps, std::move(state),
                               std::forward<TokenCollector>(collect_token), &result));
    return result;
}

template <typename ModelT, typename TokenCollector>
base::Status RunGeneration(const ModelT& model, std::vector<int32_t> tokens, int32_t total_steps,
                           GenerationState state, TokenCollector&& collect_token,
                           GenerationResult* result) {
    if (result == nullptr) {
        return base::error::InvalidArgument("The generation result pointer is empty.");
    }
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";
    const int32_t prompt_len = static_cast<int32_t>(tokens.size());
    const auto prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    int32_t pos = 0;
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        base::Status predict_status = base::error::Success();
        if (pos < prompt_len) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, true);
            predict_status = model.predict(input, pos_tensor, true, state.next);
        } else {
            tokens = std::vector<int32_t>{state.next};
            const auto token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, false);
            predict_status = model.predict(input, pos_tensor, false, state.next);
        }
        if (!predict_status.ok()) {
            return predict_status;
        }
        if (model.is_sentence_ending(state.next)) {
            break;
        }
        CollectToken(std::forward<TokenCollector>(collect_token), state, tokens, pos, prompt_len);
        ++pos;
    }

    *result = GenerationResult{std::min(pos, total_steps), std::move(state)};
    return base::error::Success();
}

template <typename ModelT, typename TokenCollector>
int32_t GenerateText(const ModelT& model, std::vector<int32_t> tokens, int total_steps,
                     GenerationState state, TokenCollector&& collect_token,
                     bool need_output = false) {
    auto result = RunGeneration(model, std::move(tokens), total_steps, std::move(state),
                                std::forward<TokenCollector>(collect_token));

    if (need_output) {
        printf("%s ", model.decode(result.state.words).data());
        fflush(stdout);
    }
    return result.steps;
}

template <typename ModelT>
base::Status GenerateGreedyText(const ModelT& model, const std::string& sentence,
                                int32_t total_steps, bool need_output, int32_t* steps) {
    if (steps == nullptr) {
        return base::error::InvalidArgument("The steps pointer is empty.");
    }

    GenerationResult result;
    const auto tokens = model.encode(sentence);
    const base::Status status =
        RunGeneration(model, tokens, total_steps, GenerationState{},
                      CollectPromptAndGeneratedTokens, &result);
    if (!status.ok()) {
        return status;
    }

    if (need_output) {
        printf("%s ", model.decode(result.state.words).data());
        fflush(stdout);
    }
    *steps = result.steps;
    return base::error::Success();
}
}  // namespace app

#endif  // KUIPER_MODELS_COMMON_GENERATION_H_
