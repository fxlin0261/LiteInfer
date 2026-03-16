#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <base/base.h>
#include <glog/logging.h>
#include "common/generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 8192;
constexpr int32_t kDefaultGenerationSteps = 128;

int32_t ParsePositiveIntArg(const char* arg_name, const char* value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed <= 0 ||
        parsed > std::numeric_limits<int32_t>::max()) {
        LOG(FATAL) << arg_name << " must be a positive integer, got: " << value;
    }
    return static_cast<int32_t>(parsed);
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        LOG(INFO) << "Usage: ./llama3_infer <checkpoint_path> <tokenizer_path> "
                  << "[runtime_max_seq_len] [steps]";
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];
    const int32_t runtime_max_seq_len =
        argc >= 4 ? ParsePositiveIntArg("runtime_max_seq_len", argv[3])
                  : kDefaultRuntimeMaxSeqLen;
    const int32_t generation_steps =
        argc >= 5 ? ParsePositiveIntArg("steps", argv[4]) : kDefaultGenerationSteps;
    model::Llama3Model model(tokenizer_path, checkpoint_path, false);
    const auto max_seq_status = model.set_runtime_max_seq_len(runtime_max_seq_len);
    if (!max_seq_status.ok()) {
        LOG(FATAL) << "Invalid runtime max seq len, code: "
                   << static_cast<int>(max_seq_status.code())
                   << ", message: " << max_seq_status.message();
    }
    LOG(INFO) << "Using runtime max seq len " << runtime_max_seq_len << " and generation steps "
              << generation_steps;
    const auto init_status = model.init(base::DefaultDeviceType());
    if (!init_status.ok()) {
        LOG(FATAL) << "The model init failed, code: " << static_cast<int>(init_status.code())
                   << ", message: " << init_status.message();
    }
    const std::string sentence = "hello";
    const auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int32_t steps = 0;
    const auto generate_status =
        app::GenerateGreedyText(model, sentence, generation_steps, true, &steps);
    if (!generate_status.ok()) {
        LOG(FATAL) << "Text generation failed, code: "
                   << static_cast<int>(generate_status.code())
                   << ", message: " << generate_status.message();
    }
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
