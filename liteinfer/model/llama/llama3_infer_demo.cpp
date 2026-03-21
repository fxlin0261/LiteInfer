#include <chrono>
#include <cstdlib>
#include <iostream>

#include <base/base.h>
#include <glog/logging.h>

#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 4096;
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./llama3_infer_demo <checkpoint_path> <tokenizer_path>";
        return EXIT_FAILURE;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    model::Llama3Model model(tokenizer_path, checkpoint_path, false);
    const auto init_status = model.init(base::DefaultDeviceType(), kDefaultRuntimeMaxSeqLen);
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }
    const int32_t runtime_max_seq_len = model.max_seq_len();
    const int32_t max_context_steps = runtime_max_seq_len;
    LOG(INFO) << "Using runtime max seq len " << runtime_max_seq_len
              << ". Generation stops automatically at EOS or when the KV cache is full.";
    const std::string sentence = "hello";
    const auto start = std::chrono::steady_clock::now();
    std::cout << "Generating...\n" << std::flush;
    int32_t executed_steps = 0;
    const auto generate_status =
        app::GenerateGreedyText(model, sentence, max_context_steps, true, &executed_steps);
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
