#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <base/base.h>
#include <glog/logging.h>

#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 4096;

bool ParseRuntimeMaxSeqLenArg(int argc, char* argv[], int32_t* runtime_max_seq_len,
                              int* first_positional_arg_index) {
    CHECK(runtime_max_seq_len != nullptr);
    CHECK(first_positional_arg_index != nullptr);

    *runtime_max_seq_len = kDefaultRuntimeMaxSeqLen;
    *first_positional_arg_index = 1;
    if (argc < 3 || std::string(argv[1]) != "--max-seq-len") {
        return true;
    }
    if (argc < 5) {
        return false;
    }

    char* parse_end = nullptr;
    const long parsed_value = std::strtol(argv[2], &parse_end, 10);
    if (parse_end == argv[2] || *parse_end != '\0' || parsed_value <= 0) {
        return false;
    }

    *runtime_max_seq_len = static_cast<int32_t>(parsed_value);
    *first_positional_arg_index = 3;
    return true;
}

std::string JoinPromptArgs(int argc, char* argv[], int start_index) {
    std::string prompt;
    for (int index = start_index; index < argc; ++index) {
        if (!prompt.empty()) {
            prompt += ' ';
        }
        prompt += argv[index];
    }
    return prompt;
}

std::vector<int32_t> GeneratedTokensOnly(const std::vector<int32_t>& all_output_tokens,
                                         int32_t prompt_token_count) {
    if (prompt_token_count >= static_cast<int32_t>(all_output_tokens.size())) {
        return {};
    }
    return std::vector<int32_t>(all_output_tokens.begin() + prompt_token_count,
                                all_output_tokens.end());
}
}  // namespace

int main(int argc, char* argv[]) {
    int32_t requested_runtime_max_seq_len = kDefaultRuntimeMaxSeqLen;
    int first_positional_arg_index = 1;
    if (!ParseRuntimeMaxSeqLenArg(argc, argv, &requested_runtime_max_seq_len,
                                  &first_positional_arg_index) ||
        argc - first_positional_arg_index < 3) {
        LOG(INFO) << "Usage: ./llama3_infer_prompt_demo [--max-seq-len <n>] "
                     "<checkpoint_path> <tokenizer_path> <prompt>";
        return EXIT_FAILURE;
    }

    const std::string checkpoint_path = argv[first_positional_arg_index];
    const std::string tokenizer_path = argv[first_positional_arg_index + 1];
    const std::string prompt = JoinPromptArgs(argc, argv, first_positional_arg_index + 2);

    model::Llama3Model model(tokenizer_path, checkpoint_path, false);
    const auto init_status =
        model.init(base::DefaultDeviceType(), requested_runtime_max_seq_len);
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }

    const int32_t runtime_max_seq_len = model.max_seq_len();
    const int32_t max_context_steps = runtime_max_seq_len;
    LOG(INFO) << "Prompt: " << prompt;
    LOG(INFO) << "Using checkpoint: " << checkpoint_path;
    LOG(INFO) << "Using tokenizer: " << tokenizer_path;
    LOG(INFO) << "Using runtime max seq len " << runtime_max_seq_len
              << ". Generation stops automatically at EOS or when the KV cache is full.";

    const auto start = std::chrono::steady_clock::now();
    std::cout << "Generating...\n" << std::flush;
    const auto tokens = model.encode(prompt);
    app::GenerationState generation_result;
    const auto generate_status =
        app::RunGeneration(model, tokens, max_context_steps, &generation_result);
    if (!generate_status.ok()) {
        LOG(FATAL) << "Text generation failed, code: "
                   << static_cast<int>(generate_status.code())
                   << ", message: " << generate_status.message();
    }

    const int32_t prompt_token_count =
        std::max<int32_t>(0, static_cast<int32_t>(tokens.size()) - 1);
    const auto generated_tokens =
        GeneratedTokensOnly(generation_result.words, prompt_token_count);
    std::cout << model.decode(generated_tokens) << ' ' << std::flush;

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << "\nsteps/s:" << (static_cast<double>(generation_result.executed_steps) / duration)
              << '\n';
    return EXIT_SUCCESS;
}
