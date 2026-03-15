#include <base/base.h>
#include <base/tick.h>
#include <glog/logging.h>
#include "model/llama/llama.h"

int32_t generate(const model::Llama2Model& model, const std::string& sentence, int total_steps,
                 bool need_output = false) {
    // llama2 加了BOS 没加EOS
    // 返回一串tokens id: BOS+id
    auto tokens = model.encode(sentence);
    int32_t prompt_len = tokens.size();
    LOG_IF(FATAL, tokens.empty()) << "The tokens is empty.";

    int32_t pos = 0;
    int32_t next = -1;
    bool is_prompt = true;
    // 返回值为op::EmbeddingOutput
    // 也就是说：
    // input_tokens
    // 输入的 token id，shape 是 [token_num]
    // input_embeddings
    // embedding 查表后的结果，shape 是 [token_num, dim]
    // input_token_num
    // 表示 token 数量的 tensor，shape 是 [token_num]
    const auto& prompt_embedding = model.embedding(tokens);
    tensor::Tensor pos_tensor = model.get_buffer(model::ModelBufferType::kInputPos);

    std::vector<int32_t> words;
    while (pos < total_steps) {
        pos_tensor.index<int32_t>(0) = pos;
        if (pos < prompt_len - 1) {
            // 把当前步需要的输入向量取出来给 是个一维的
            tensor::Tensor input = model.fill_input(pos_tensor, prompt_embedding, is_prompt);
            model.predict(input, pos_tensor, is_prompt, next);
        } else {
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
    const char* checkpoint_path = argv[1];
    const char* tokenizer_path = argv[2];

    model::Llama2Model model(tokenizer_path, checkpoint_path, false);
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
