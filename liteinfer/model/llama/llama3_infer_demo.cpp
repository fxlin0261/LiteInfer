#include <chrono>
#include <cstdlib>
#include <iostream>

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
    if (argc != 5) {
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
}  // namespace

int main(int argc, char* argv[]) {
    int32_t requested_runtime_max_seq_len = kDefaultRuntimeMaxSeqLen;
    int first_positional_arg_index = 1;
    if (!ParseRuntimeMaxSeqLenArg(argc, argv, &requested_runtime_max_seq_len,
                                  &first_positional_arg_index) ||
        argc - first_positional_arg_index != 2) {
        LOG(INFO) << "Usage: ./llama3_infer_demo [--max-seq-len <n>] "
                     "<checkpoint_path> <tokenizer_path>";
        return EXIT_FAILURE;
    }
    const char* checkpoint_path = argv[first_positional_arg_index];
    const char* tokenizer_path = argv[first_positional_arg_index + 1];
    model::Llama3Model model(tokenizer_path, checkpoint_path, false);
    const auto init_status = model.init(base::DefaultDeviceType(), requested_runtime_max_seq_len);
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
