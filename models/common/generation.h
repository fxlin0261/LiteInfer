#ifndef KUIPER_MODELS_COMMON_GENERATION_H_
#define KUIPER_MODELS_COMMON_GENERATION_H_

#include <glog/logging.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
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

template <typename ModelT, typename TokenCollector>
GenerationResult RunGeneration(const ModelT& model, std::vector<int32_t> tokens, int total_steps,
                               GenerationState state, TokenCollector&& collect_token) {
    const int32_t prompt_len = static_cast<int32_t>(tokens.size());
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    int32_t pos = 0;
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        if (pos < prompt_len - 1) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, state.is_prompt);
            model.predict(input, pos_tensor, state.is_prompt, state.next);
        } else {
            state.is_prompt = false;
            tokens = std::vector<int32_t>{state.next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, state.is_prompt);
            model.predict(input, pos_tensor, state.is_prompt, state.next);
        }
        if (model.is_sentence_ending(state.next)) {
            break;
        }
        collect_token(state, tokens, pos);
        ++pos;
    }

    return GenerationResult{std::min(pos, total_steps), std::move(state)};
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

}  // namespace app

#endif  // KUIPER_MODELS_COMMON_GENERATION_H_
