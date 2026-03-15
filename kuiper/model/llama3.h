#ifndef KUIPER_INCLUDE_MODEL_LLAMA3_H_
#define KUIPER_INCLUDE_MODEL_LLAMA3_H_

#include "llama.h"

namespace model {

class Llama3Model : public LlamaModelBase {
public:
    explicit Llama3Model(std::string token_path, std::string model_path, bool is_quant_model)
        : LlamaModelBase(base::TokenizerType::kEncodeBpe, base::ModelType::kModelTypeLlama3,
                         std::move(token_path), std::move(model_path), is_quant_model) {}
};

}  // namespace model

#endif
