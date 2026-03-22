#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ctime>
#include <string>
#include <string_view>
#include <vector>

#include <base/base.h>
#include <glog/logging.h>

#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 4096;

bool ContainsCaseInsensitive(std::string_view haystack, std::string_view needle) {
    return std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                       [](unsigned char lhs, unsigned char rhs) {
                           return std::tolower(lhs) == std::tolower(rhs);
                       }) != haystack.end();
}

bool LooksLikeInstructCheckpoint(const std::string& checkpoint_path,
                                 const std::string& tokenizer_path) {
    return ContainsCaseInsensitive(checkpoint_path, "instruct") ||
           ContainsCaseInsensitive(tokenizer_path, "instruct");
}

std::string ApplyLlama3ChatTemplate(const std::string& prompt) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm {};
#if defined(_WIN32)
    localtime_s(&local_tm, &now_time);
#else
    localtime_r(&now_time, &local_tm);
#endif

    static constexpr const char* kMonths[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    CHECK_GE(local_tm.tm_mon, 0);
    CHECK_LT(local_tm.tm_mon, 12);

    char date_buf[32];
    std::snprintf(date_buf, sizeof(date_buf), "%02d %s %04d", local_tm.tm_mday,
                  kMonths[local_tm.tm_mon], local_tm.tm_year + 1900);

    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
           "Cutting Knowledge Date: December 2023\n"
           "Today Date: " +
           std::string(date_buf) +
           "\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + prompt +
           "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
}

bool ParseArgs(int argc, char* argv[], int32_t* runtime_max_seq_len, bool* use_raw_prompt,
               int* first_positional_arg_index) {
    CHECK(runtime_max_seq_len != nullptr);
    CHECK(use_raw_prompt != nullptr);
    CHECK(first_positional_arg_index != nullptr);

    *runtime_max_seq_len = kDefaultRuntimeMaxSeqLen;
    *use_raw_prompt = false;
    *first_positional_arg_index = 1;

    int arg_index = 1;
    while (arg_index < argc) {
        const std::string_view arg = argv[arg_index];
        if (arg == "--max-seq-len") {
            if (arg_index + 1 >= argc) {
                return false;
            }
            char* parse_end = nullptr;
            const long parsed_value = std::strtol(argv[arg_index + 1], &parse_end, 10);
            if (parse_end == argv[arg_index + 1] || *parse_end != '\0' || parsed_value <= 0) {
                return false;
            }
            *runtime_max_seq_len = static_cast<int32_t>(parsed_value);
            arg_index += 2;
            continue;
        }
        if (arg == "--raw-prompt") {
            *use_raw_prompt = true;
            ++arg_index;
            continue;
        }
        break;
    }

    *first_positional_arg_index = arg_index;
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

bool DebugPromptTokensEnabled() {
    const char* env = std::getenv("LITEINFER_DEBUG_PROMPT_TOKENS");
    return env != nullptr && std::strcmp(env, "0") != 0;
}

std::string SanitizeDebugText(std::string text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        switch (ch) {
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '"':
                out += "\\\"";
                break;
            default:
                if (ch >= 0x20 && ch <= 0x7E) {
                    out.push_back(static_cast<char>(ch));
                } else {
                    char buf[5];
                    std::snprintf(buf, sizeof(buf), "\\x%02X", ch);
                    out += buf;
                }
                break;
        }
    }
    return out;
}

void LogPromptTokens(const model::Model& model, const std::string& formatted_prompt,
                     const std::vector<int32_t>& tokens) {
    LOG(INFO) << "Formatted prompt: \"" << SanitizeDebugText(formatted_prompt) << "\"";
    LOG(INFO) << "Prompt token count: " << tokens.size();
    for (size_t index = 0; index < tokens.size(); ++index) {
        const int32_t token_id = tokens[index];
        LOG(INFO) << "  prompt_token[" << index << "] id=" << token_id << " text=\""
                  << SanitizeDebugText(model.decode(token_id)) << "\"";
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    int32_t requested_runtime_max_seq_len = kDefaultRuntimeMaxSeqLen;
    bool use_raw_prompt = false;
    int first_positional_arg_index = 1;
    if (!ParseArgs(argc, argv, &requested_runtime_max_seq_len, &use_raw_prompt,
                                  &first_positional_arg_index) ||
        argc - first_positional_arg_index < 3) {
        LOG(INFO) << "Usage: ./llama3_infer_prompt_demo [--max-seq-len <n>] [--raw-prompt] "
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
    const bool should_apply_chat_template =
        !use_raw_prompt && LooksLikeInstructCheckpoint(checkpoint_path, tokenizer_path);
    if (should_apply_chat_template) {
        LOG(INFO) << "Applying the Llama 3 instruct chat template.";
    } else if (use_raw_prompt) {
        LOG(INFO) << "Using the raw prompt without a chat template.";
    }
    const std::string formatted_prompt =
        should_apply_chat_template ? ApplyLlama3ChatTemplate(prompt) : prompt;
    const auto tokens = model.encode(formatted_prompt);
    if (DebugPromptTokensEnabled()) {
        LogPromptTokens(model, formatted_prompt, tokens);
    }
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
