#ifndef KUIPER_INCLUDE_MODEL_LLAMA_H_
#define KUIPER_INCLUDE_MODEL_LLAMA_H_

#include "model/decoder/standard_decoder.h"

namespace model {
class LlamaModelBase : public StandardDecoderModel {
public:
    explicit LlamaModelBase(base::TokenizerType tokenizer_type, base::ModelType model_type,
                            std::string token_path, std::string model_path, bool is_quant_model);

protected:
    base::Status create_param_layers() override;
    base::Status create_param_quant_layers() override;
};

class Llama2Model : public LlamaModelBase {
public:
    explicit Llama2Model(std::string token_path, std::string model_path, bool is_quant_model)
        : LlamaModelBase(base::TokenizerType::kEncodeSpe, base::ModelType::kModelTypeLlama2,
                         std::move(token_path), std::move(model_path), is_quant_model) {}
};

class Llama3Model : public LlamaModelBase {
public:
    explicit Llama3Model(std::string token_path, std::string model_path, bool is_quant_model)
        : LlamaModelBase(base::TokenizerType::kEncodeBpe, base::ModelType::kModelTypeLlama3,
                         std::move(token_path), std::move(model_path), is_quant_model) {}
};
}  // namespace model
#endif
