#ifndef LITEINFER_BASE_SAMPLER_H_
#define LITEINFER_BASE_SAMPLER_H_
#include <cstddef>
#include <cstdint>
#include "base/base.h"

namespace sampler {
class Sampler {
public:
    explicit Sampler(base::DeviceType device_type) : device_type_(device_type) {}
    virtual ~Sampler() = default;
    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;
    virtual bool requires_host_logits(base::DeviceType logits_device_type) const {
        UNUSED(logits_device_type);
        return false;
    }

protected:
    base::DeviceType device_type_;
};
}  // namespace sampler
#endif  // LITEINFER_BASE_SAMPLER_H_
