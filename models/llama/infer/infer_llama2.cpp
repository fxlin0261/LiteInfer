#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <base/base.h>
#include <glog/logging.h>
#include "generation.h"
#include "model/llama/llama.h"

namespace {
constexpr int32_t kDefaultRuntimeMaxSeqLen = 8192;
constexpr int32_t kDefaultMaxTotalSteps = 128;

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
        LOG(INFO) << "Usage: ./llama2_infer <checkpoint_path> <tokenizer_path> "
                  << "[runtime_max_seq_len] [max_total_steps]";
        return -1;
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
    printf("Generating...\n");
    fflush(stdout);
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
    printf("\nsteps/s:%lf\n", static_cast<double>(executed_steps) / duration);
    fflush(stdout);
    return 0;
}
