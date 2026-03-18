#ifndef LITEINFER_BASE_ARGMAX_SAMPLER_H_
#define LITEINFER_BASE_ARGMAX_SAMPLER_H_
#include <base/base.h>
#include "base/sampler.h"

namespace sampler {
class ArgmaxSampler : public Sampler {
public:
    explicit ArgmaxSampler(base::DeviceType device_type) : Sampler(device_type) {}
    size_t sample(const float* logits, size_t size, void* stream = nullptr) override;
};
}  // namespace sampler
#endif  // LITEINFER_BASE_ARGMAX_SAMPLER_H_
