#ifndef TOPK_SAMPLING_KERNEL_CUH
#define TOPK_SAMPLING_KERNEL_CUH

#include <cstddef>

namespace kernel {
size_t topk_sampling_kernel_cu(const float* logits, size_t size, size_t top_k, float temperature,
                               double random_value, void* stream = nullptr);
}

#endif  // TOPK_SAMPLING_KERNEL_CUH
