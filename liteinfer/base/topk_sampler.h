#ifndef LITEINFER_BASE_TOPK_SAMPLER_H_
#define LITEINFER_BASE_TOPK_SAMPLER_H_

#include <cstdint>
#include <random>
#include "base/sampler.h"

namespace sampler {
class TopKSampler : public Sampler {
public:
    explicit TopKSampler(base::DeviceType device_type, size_t top_k = 40, float temperature = 0.8f,
                         uint32_t seed = 0);
    size_t sample(const float* logits, size_t size, void* stream = nullptr) override;
    bool requires_host_logits(base::DeviceType logits_device_type) const override {
        return logits_device_type == base::DeviceType::kDeviceCUDA;
    }

private:
    size_t top_k_ = 40;
    float temperature_ = 0.8f;
    std::mt19937 rng_;
};
}  // namespace sampler

#endif  // LITEINFER_BASE_TOPK_SAMPLER_H_
