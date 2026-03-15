#ifndef KUIPER_INCLUDE_MODEL_LLAMA2_H_
#define KUIPER_INCLUDE_MODEL_LLAMA2_H_

#include "llama.h"

namespace model {

class Llama2Model : public LlamaModelBase {
public:
    explicit Llama2Model(std::string token_path, std::string model_path, bool is_quant_model)
        : LlamaModelBase(base::TokenizerType::kEncodeSpe, base::ModelType::kModelTypeLlama2,
                         std::move(token_path), std::move(model_path), is_quant_model) {}
};
}  // namespace model

#endif
