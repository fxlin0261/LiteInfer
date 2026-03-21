#ifndef TOPK_SAMPLING_KERNEL_H
#define TOPK_SAMPLING_KERNEL_H

#include <cstddef>

namespace kernel {
size_t topk_sampling_kernel_cpu(const float* logits, size_t size, size_t top_k, float temperature,
                                double random_value, void* stream = nullptr);
}

#endif  // TOPK_SAMPLING_KERNEL_H
