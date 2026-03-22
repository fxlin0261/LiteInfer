#ifndef LITEINFER_INCLUDE_MODEL_LLAMA_H_
#define LITEINFER_INCLUDE_MODEL_LLAMA_H_

#include "model/llama/llama_decoder.h"

namespace model {
class LlamaModel : public LlamaDecoderModel {
public:
    explicit LlamaModel(base::TokenizerType tokenizer_type, base::ModelType model_type,
                        std::string token_path, std::string model_path, bool is_quant_model);

protected:
    base::Status create_param_layers() override;
    base::Status create_param_quant_layers() override;
};

class Llama3Model : public LlamaModel {
public:
    explicit Llama3Model(std::string token_path, std::string model_path, bool is_quant_model)
        : LlamaModel(base::TokenizerType::kEncodeBpe, base::ModelType::kModelTypeLlama3,
                         std::move(token_path), std::move(model_path), is_quant_model) {}
};
}  // namespace model
#endif
