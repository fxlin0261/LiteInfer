#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "common/generation.h"
#include "model/qwen/qwen2.h"

int32_t generate(const model::Qwen2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    auto tokens = model.encode(sentence);
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    app::GenerationState state;
    state.next = tokens.front();
    state.words.push_back(state.next);
    return app::GenerateText(
        model, std::move(tokens), total_steps, std::move(state),
        [](app::GenerationState& state, std::vector<int32_t>& tokens, int32_t pos) {
            if (state.is_prompt) {
                state.next = tokens.at(pos + 1);
            }
            state.words.push_back(state.next);
        },
        need_output);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];

    model::Qwen2Model model(tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DefaultDeviceType());
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }
    const std::string& sentence = "hi!";

    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int steps = generate(model, sentence, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps:%d\n", steps);
    printf("\nduration:%lf\n", duration);
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
