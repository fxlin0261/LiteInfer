#ifndef LITEINFER_MODEL_GENERATION_H_
#define LITEINFER_MODEL_GENERATION_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "model/model.h"

namespace app {
struct GenerationState {
    int32_t next = -1;
    bool is_prompt = true;
    int32_t executed_steps = 0;
    std::vector<int32_t> words;
};

template <typename ModelT>
base::Status RunGeneration(const ModelT& model, std::vector<int32_t> tokens,
                           int32_t max_total_steps, GenerationState* result) {
    if (result == nullptr) {
        return base::error::InvalidArgument("The generation result pointer is empty.");
    }
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    GenerationState state;
    const int32_t prompt_len = static_cast<int32_t>(tokens.size());
    const auto prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_runtime_tensor(model::RuntimeTensorType::kInputPos);
    int32_t pos = 0;
    while (pos < max_total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        base::Status predict_status = base::error::Success();
        if (pos < prompt_len) {
            const bool is_prompt_step = pos < prompt_len - 1;
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, true);
            predict_status = model.predict(input, pos_tensor, is_prompt_step, state.next);
        } else {
            tokens = {state.next};
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

        if (pos < prompt_len - 1) {
            // “下一个要用的 token”不是模型自由生成的结果，而是 prompt 里本来就存在的下一个 token
            state.next = tokens.at(pos + 1);
        } else {
            //  prompt 已经走到最后一个 token 了从下一步开始就不再靠 prompt 往后推
            state.is_prompt = false;
        }
        state.words.push_back(state.next);
        ++pos;
    }

    state.executed_steps = pos;
    *result = std::move(state);
    return base::error::Success();
}

template <typename ModelT>
base::Status GenerateGreedyText(const ModelT& model, const std::string& sentence,
                                int32_t max_total_steps, bool need_output,
                                int32_t* executed_steps) {
    if (executed_steps == nullptr) {
        return base::error::InvalidArgument("The executed steps pointer is empty.");
    }

    GenerationState result;
    const auto tokens = model.encode(sentence);
    const base::Status status = RunGeneration(model, tokens, max_total_steps, &result);
    if (!status.ok()) {
        return status;
    }

    if (need_output) {
        std::cout << model.decode(result.words) << ' ' << std::flush;
    }
    *executed_steps = result.executed_steps;
    return base::error::Success();
}
}  // namespace app

#endif  // LITEINFER_MODEL_GENERATION_H_
