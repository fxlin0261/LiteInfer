#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama/llama.h"

int32_t generate(const model::Llama3Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    // 这里先把文本 sentence 编码成 token id 序列。prompt_len 是提示词长度。如果一个 token
    // 都没有，程序直接报错退出
    auto tokens = model.encode(sentence);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    // pos：当前处理到第几个位置
    // next：模型当前预测出的下一个 token
    // is_prompt：现在是不是还处于“喂提示词”阶段
    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);
    std::vector<int32_t> words;

    //  pos 像“当前走到第几格”
    //  total_steps 像“最多只能走多少格”
    //  pos < total_steps 就是“还没走满，可以继续”
    while (pos < total_steps) {
        // 第0个位置赋值
        pos_tensor.index<int32_t>(0) = pos;
        // 不是最后一个
        if (pos < prompt_len - 1) {
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {  // 最后一个
            is_prompt = false;
            tokens = std::vector<int32_t>{next};
            const auto& token_embedding = model.embedding(tokens);
            tensor::Tensor input = model.fill_input(pos_tensor, token_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        }
        if (model.is_sentence_ending(next)) {
            break;
        }
        if (is_prompt) {
            next = tokens.at(pos + 1);
            words.push_back(next);
        } else {
            words.push_back(next);
        }

        pos += 1;
    }
    if (need_output) {
        printf("%s ", model.decode(words).data());
        fflush(stdout);
    }
    return std::min(pos, total_steps);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
        return -1;
    }
    const char* checkpoint_path = argv[1];  // e.g. out/model.bin
    const char* tokenizer_path = argv[2];

    model::Llama3Model model(tokenizer_path, checkpoint_path, false);
    auto init_status = model.init(base::DefaultDeviceType());
    if (!init_status) {
        LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
    }
    const std::string& sentence = "hello";

    auto start = std::chrono::steady_clock::now();
    printf("Generating...\n");
    fflush(stdout);
    int steps = generate(model, sentence, 128, true);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(end - start).count();
    printf("\nsteps/s:%lf\n", static_cast<double>(steps) / duration);
    fflush(stdout);
    return 0;
}
