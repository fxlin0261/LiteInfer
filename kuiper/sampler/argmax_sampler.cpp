#include "sampler/argmax_sampler.h"
#include <algorithm>
#if KUIPER_ENABLE_CUDA
#include "../op/kernels/cuda/argmax_kernel.cuh"
#endif

namespace sampler {
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
  if (device_type_ == base::DeviceType::kDeviceCPU) {
    size_t next = std::distance(logits, std::max_element(logits, logits + size));
    return next;
  } else {
#if KUIPER_ENABLE_CUDA
    size_t next = kernel::argmax_kernel_cu(logits, size, stream);
    return next;
#else
    LOG(FATAL) << "CUDA sampling requested in a CPU-only build.";
    return 0;
#endif
  }
}
}  // namespace sampler
