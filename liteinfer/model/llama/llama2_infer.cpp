#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <base/base.h>
#include <glog/logging.h>
#include "model/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 8192;
constexpr int32_t kDefaultMaxTotalSteps = 128;

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
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        LOG(INFO) << "Usage: ./llama2_infer <checkpoint_path> <tokenizer_path> "
                  << "[runtime_max_seq_len] [max_total_steps]";
        return EXIT_FAILURE;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    const int32_t runtime_max_seq_len =
        argc >= 4 ? ParsePositiveIntArg("runtime_max_seq_len", argv[3])
                  : kDefaultRuntimeMaxSeqLen;
    const int32_t max_total_steps =
        argc >= 5 ? ParsePositiveIntArg("max_total_steps", argv[4]) : kDefaultMaxTotalSteps;
    model::Llama2Model model(tokenizer_path, checkpoint_path, false);
    LOG(INFO) << "Using runtime max seq len " << runtime_max_seq_len << " and max total steps "
              << max_total_steps;
    const auto init_status = model.init(base::DefaultDeviceType(), runtime_max_seq_len);
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }
    const std::string sentence = "hello";
    const auto start = std::chrono::steady_clock::now();
    std::cout << "Generating...\n" << std::flush;
    int32_t executed_steps = 0;
    const auto generate_status =
        app::GenerateGreedyText(model, sentence, max_total_steps, true, &executed_steps);
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
