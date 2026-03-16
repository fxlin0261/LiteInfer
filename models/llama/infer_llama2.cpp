#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "common/generation.h"
#include "model/llama/llama.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        return -1;
    }
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    model::Llama2Model model(tokenizer_path, checkpoint_path, false);
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
    const auto generate_status = app::GenerateGreedyText(model, sentence, 128, true, &steps);
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
