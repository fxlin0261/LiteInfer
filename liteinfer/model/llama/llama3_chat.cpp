#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <base/base.h>
#include <glog/logging.h>

#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 8192;
constexpr int32_t kDefaultMaxNewTokens = 128;
constexpr std::string_view kDefaultSystemPrompt = "You are a concise and helpful AI assistant.";

struct ChatTurn {
    std::string user;
    std::string assistant;
};

int32_t ParsePositiveIntArg(std::string_view arg_name, const char* value) {
    const std::string_view value_view(value);
    int32_t parsed = 0;
    const auto [end, error] =
        std::from_chars(value_view.data(), value_view.data() + value_view.size(), parsed);
    if (error != std::errc{} || end != value_view.data() + value_view.size() || parsed <= 0) {
        LOG(FATAL) << arg_name << " must be a positive integer, got: " << value;
    }
    return parsed;
}

std::string_view TrimAsciiWhitespace(std::string_view text) {
    while (!text.empty() &&
           (text.front() == ' ' || text.front() == '\t' || text.front() == '\r' ||
            text.front() == '\n')) {
        text.remove_prefix(1);
    }
    while (!text.empty() &&
           (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' ||
            text.back() == '\n')) {
        text.remove_suffix(1);
    }
    return text;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

void PrintHelp() {
    std::cout << "Commands:\n"
              << "  /help                Show this help message.\n"
              << "  /clear               Clear the current conversation history.\n"
              << "  /system <prompt>     Replace the system prompt and clear history.\n"
              << "  /quit or /exit       Exit the chat.\n";
}

void AppendLlama3Message(std::string_view role, std::string_view content, std::string* prompt) {
    CHECK(prompt != nullptr);
    prompt->append("<|start_header_id|>");
    prompt->append(role);
    prompt->append("<|end_header_id|>\n\n");
    prompt->append(content);
    prompt->append("<|eot_id|>");
}

std::string BuildPrompt(std::string_view system_prompt, const std::vector<ChatTurn>& history,
                        std::string_view user_input) {
    std::string prompt;
    AppendLlama3Message("system", system_prompt, &prompt);
    for (const auto& turn : history) {
        AppendLlama3Message("user", turn.user, &prompt);
        AppendLlama3Message("assistant", turn.assistant, &prompt);
    }
    AppendLlama3Message("user", user_input, &prompt);
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return prompt;
}

bool PreparePromptThatFits(const model::Llama3Model& model, std::string_view system_prompt,
                           std::vector<ChatTurn>* history, std::string_view user_input,
                           int32_t max_new_tokens, int32_t runtime_max_seq_len,
                           std::vector<int32_t>* prompt_tokens, bool* history_trimmed) {
    CHECK(history != nullptr);
    CHECK(prompt_tokens != nullptr);
    CHECK(history_trimmed != nullptr);

    *history_trimmed = false;
    const size_t max_seq_len = static_cast<size_t>(runtime_max_seq_len);
    const size_t new_token_budget = static_cast<size_t>(max_new_tokens);
    while (true) {
        *prompt_tokens = model.encode(BuildPrompt(system_prompt, *history, user_input));
        if (prompt_tokens->size() < max_seq_len &&
            prompt_tokens->size() + new_token_budget <= max_seq_len) {
            return true;
        }
        if (history->empty()) {
            return false;
        }
        history->erase(history->begin());
        *history_trimmed = true;
    }
}

base::Status GenerateAssistantReply(const model::Llama3Model& model,
                                    const std::vector<int32_t>& prompt_tokens,
                                    int32_t max_new_tokens, int32_t runtime_max_seq_len,
                                    std::string* reply, int32_t* generated_tokens) {
    if (reply == nullptr || generated_tokens == nullptr) {
        return base::error::InvalidArgument("The reply output pointers must not be null.");
    }
    if (prompt_tokens.empty()) {
        return base::error::InvalidArgument("The prompt tokens are empty.");
    }

    const int32_t prompt_len = static_cast<int32_t>(prompt_tokens.size());
    const int32_t available_new_tokens =
        std::max<int32_t>(0, runtime_max_seq_len - prompt_len);
    if (available_new_tokens <= 0) {
        return base::error::InvalidArgument("The prompt is longer than the runtime sequence limit.");
    }

    const int32_t capped_max_new_tokens = std::min(max_new_tokens, available_new_tokens);
    const int32_t max_total_steps = prompt_len + capped_max_new_tokens;
    app::GenerationState generation_state;
    const auto status = app::RunGeneration(model, prompt_tokens, max_total_steps, &generation_state);
    if (!status.ok()) {
        return status;
    }

    const int32_t generated_token_start = std::max<int32_t>(0, prompt_len - 1);
    std::vector<int32_t> response_tokens;
    if (generated_token_start < static_cast<int32_t>(generation_state.words.size())) {
        response_tokens.assign(generation_state.words.begin() + generated_token_start,
                               generation_state.words.end());
    }

    *generated_tokens = static_cast<int32_t>(response_tokens.size());
    *reply = model.decode(response_tokens);
    return base::error::Success();
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        LOG(INFO) << "Usage: ./llama3_chat <checkpoint_path> <tokenizer_path> "
                  << "[runtime_max_seq_len] [max_new_tokens]";
        return EXIT_FAILURE;
    }

    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    const int32_t requested_runtime_max_seq_len =
        argc >= 4 ? ParsePositiveIntArg("runtime_max_seq_len", argv[3])
                  : kDefaultRuntimeMaxSeqLen;
    const int32_t max_new_tokens =
        argc >= 5 ? ParsePositiveIntArg("max_new_tokens", argv[4]) : kDefaultMaxNewTokens;

    model::Llama3Model model(tokenizer_path, checkpoint_path, false);
    const auto init_status = model.init(base::DefaultDeviceType(), requested_runtime_max_seq_len);
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }

    const int32_t runtime_max_seq_len = model.max_seq_len();
    const int32_t effective_max_new_tokens =
        std::min(max_new_tokens, std::max<int32_t>(1, runtime_max_seq_len - 1));
    LOG(INFO) << "Interactive chat ready. Using runtime max seq len " << runtime_max_seq_len
              << " and max new tokens " << effective_max_new_tokens;

    std::cout << "LiteInfer Llama3 chat\n"
              << "Type /help for commands, /clear to reset history, /quit to exit.\n";

    std::string system_prompt(kDefaultSystemPrompt);
    std::vector<ChatTurn> history;
    std::string user_input;
    while (true) {
        std::cout << "\nYou: " << std::flush;
        if (!std::getline(std::cin, user_input)) {
            std::cout << '\n';
            break;
        }

        const std::string_view trimmed_input = TrimAsciiWhitespace(user_input);
        if (trimmed_input.empty()) {
            continue;
        }
        if (trimmed_input == "/quit" || trimmed_input == "/exit") {
            break;
        }
        if (trimmed_input == "/help") {
            PrintHelp();
            continue;
        }
        if (trimmed_input == "/clear") {
            history.clear();
            std::cout << "Conversation history cleared.\n";
            continue;
        }
        if (trimmed_input == "/system" || StartsWith(trimmed_input, "/system ")) {
            const std::string_view system_value =
                TrimAsciiWhitespace(trimmed_input.substr(std::string_view("/system").size()));
            if (system_value.empty()) {
                std::cout << "Current system prompt: " << system_prompt << '\n';
                continue;
            }
            system_prompt.assign(system_value.begin(), system_value.end());
            history.clear();
            std::cout << "System prompt updated and history cleared.\n";
            continue;
        }

        std::vector<int32_t> prompt_tokens;
        bool history_trimmed = false;
        if (!PreparePromptThatFits(model, system_prompt, &history, trimmed_input,
                                   effective_max_new_tokens, runtime_max_seq_len, &prompt_tokens,
                                   &history_trimmed)) {
            std::cout << "The prompt is too long even after dropping all history. "
                         "Please shorten the message or reduce the system prompt.\n";
            continue;
        }

        if (history_trimmed) {
            std::cout << "[History trimmed to fit the context window]\n";
        }

        std::cout << "Assistant: " << std::flush;
        const auto start = std::chrono::steady_clock::now();
        std::string reply;
        int32_t generated_tokens = 0;
        const auto status = GenerateAssistantReply(model, prompt_tokens, effective_max_new_tokens,
                                                   runtime_max_seq_len, &reply, &generated_tokens);
        const auto end = std::chrono::steady_clock::now();
        if (!status.ok()) {
            std::cout << "\nGeneration failed, code: " << static_cast<int>(status.code())
                      << ", message: " << status.message() << '\n';
            continue;
        }

        if (generated_tokens == 0) {
            std::cout << "(empty response)";
        } else {
            std::cout << reply;
        }
        std::cout << '\n';

        history.push_back(
            ChatTurn{std::string(trimmed_input.begin(), trimmed_input.end()), std::move(reply)});

        const auto duration_seconds = std::chrono::duration<double>(end - start).count();
        if (duration_seconds > 0.0 && generated_tokens > 0) {
            std::cout << "Generated " << generated_tokens << " tokens at "
                      << (static_cast<double>(generated_tokens) / duration_seconds) << " tok/s\n";
        }
    }

    return EXIT_SUCCESS;
}
