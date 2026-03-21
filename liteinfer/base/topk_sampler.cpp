#include "base/topk_sampler.h"

#include "op/kernels/kernels_interface.h"

namespace sampler {
TopKSampler::TopKSampler(base::DeviceType device_type, size_t top_k, float temperature,
                         uint32_t seed)
    : Sampler(device_type),
      top_k_(top_k),
      temperature_(temperature),
      rng_(seed == 0 ? std::random_device{}() : seed) {}

size_t TopKSampler::sample(const float* logits, size_t size, void* stream) {
    CHECK_NE(logits, nullptr) << "The logits pointer is null.";
    CHECK_GT(size, 0U) << "The logits size must be positive.";
    CHECK_GT(top_k_, 0U) << "Top-k sampling requires top_k > 0.";
    CHECK_GT(temperature_, 0.f) << "Top-k sampling requires temperature > 0.";
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const auto sampling_kernel = kernel::get_topk_sampling_kernel(device_type_);
    return sampling_kernel(logits, size, top_k_, temperature_, distribution(rng_), stream);
}
}  // namespace sampler
