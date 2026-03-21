#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <base/base.h>
#include <glog/logging.h>

#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 8192;
constexpr const char* kCheckpointRelativePath = "local_models/llama3/Llama-3.2-1B.bin";
constexpr const char* kTokenizerRelativePath = "local_models/llama3/Llama-3.2-1B/tokenizer.json";

std::string JoinPromptArgs(int argc, char* argv[]) {
    std::string prompt;
    for (int index = 1; index < argc; ++index) {
        if (!prompt.empty()) {
            prompt += ' ';
        }
        prompt += argv[index];
    }
    return prompt;
}

std::filesystem::path ResolveProjectRoot(const char* executable_path) {
    return std::filesystem::absolute(executable_path).parent_path().parent_path();
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        LOG(INFO) << "Usage: ./llama3_infer_prompt_demo <prompt>";
        return EXIT_FAILURE;
    }

    const std::string prompt = JoinPromptArgs(argc, argv);
    const std::filesystem::path project_root = ResolveProjectRoot(argv[0]);
    const std::filesystem::path checkpoint_path = project_root / kCheckpointRelativePath;
    const std::filesystem::path tokenizer_path = project_root / kTokenizerRelativePath;

    if (!std::filesystem::exists(checkpoint_path)) {
        LOG(FATAL) << "Checkpoint not found: " << checkpoint_path.string();
    }
    if (!std::filesystem::exists(tokenizer_path)) {
        LOG(FATAL) << "Tokenizer not found: " << tokenizer_path.string();
    }

    LOG(INFO) << "Prompt: " << prompt;
    LOG(INFO) << "Using checkpoint: " << checkpoint_path.string();
    LOG(INFO) << "Using tokenizer: " << tokenizer_path.string();

    model::Llama3Model model(tokenizer_path.string(), checkpoint_path.string(), false);
    const auto init_status = model.init(base::DefaultDeviceType(), kDefaultRuntimeMaxSeqLen);
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }

    const int32_t runtime_max_seq_len = model.max_seq_len();
    const int32_t max_context_steps = runtime_max_seq_len;
    LOG(INFO) << "Using runtime max seq len " << runtime_max_seq_len
              << ". Generation stops automatically at EOS or when the KV cache is full.";

    const auto start = std::chrono::steady_clock::now();
    std::cout << "Generating...\n" << std::flush;
    int32_t executed_steps = 0;
    const auto generate_status =
        app::GenerateGreedyText(model, prompt, max_context_steps, true, &executed_steps);
    if (!generate_status.ok()) {
        LOG(FATAL) << "Text generation failed, code: "
                   << static_cast<int>(generate_status.code())
                   << ", message: " << generate_status.message();
    }

    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    std::cout << "\nsteps/s:" << (static_cast<double>(executed_steps) / duration) << '\n';
    return EXIT_SUCCESS;
}
