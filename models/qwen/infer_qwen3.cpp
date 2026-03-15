#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "common/generation.h"
#include "model/qwen3.h"

int32_t generate(const model::Qwen3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    auto tokens = model.encode(sentence);
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    app::GenerationState state;
    state.next = tokens.front();
    return app::GenerateText(
        model, std::move(tokens), total_steps, std::move(state),
        [](app::GenerationState& state, std::vector<int32_t>& tokens, int32_t pos) {
            if (state.is_prompt) {
                state.next = tokens.at(pos + 1);
                return;
            }
            if (state.next != 151645 && state.next != 151644) {
                state.words.push_back(state.next);
            }
        },
        need_output);
}

std::string fill_template(const std::string& content) {
    const std::string format = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n";
    std::string result = format;
    size_t pos = result.find("%s");
    if (pos != std::string::npos) {
        result.replace(pos, 2, content);
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];

    model::Qwen3Model model(tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DefaultDeviceType());
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }

    std::string hi = "What is AI?";
    std::cout << hi << "\n";
    const std::string& sentence = fill_template(hi);
    auto start = std::chrono::steady_clock::now();
    fflush(stdout);
    int steps = generate(model, sentence, 2560, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps:%d\n", steps);
    printf("\nduration:%lf\n", duration);
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
