#ifndef KUIPER_INCLUDE_MODEL_QWEN2_H_
#define KUIPER_INCLUDE_MODEL_QWEN2_H_

#include "standard_decoder.h"

namespace model {

using Qwen2Layers = StandardDecoderLayers;

class Qwen2Model : public StandardDecoderModel {
public:
    explicit Qwen2Model(std::string token_path, std::string model_path, bool is_quant_model);

private:
    base::Status create_param_layers() override;

    bool use_qwen_tokenizer() const override;
};

}  // namespace model

#endif  // KUIPER_INCLUDE_MODEL_QWEN2_H_
