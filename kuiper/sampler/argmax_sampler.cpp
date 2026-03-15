#include "sampler/argmax_sampler.h"
#include <algorithm>
#if KUIPER_ENABLE_CUDA
#include "../op/kernels/cuda/argmax_kernel.cuh"
#endif

namespace sampler {

namespace {

size_t sample_cuda_argmax(const float* logits, size_t size, void* stream) {
#if KUIPER_ENABLE_CUDA
    return kernel::argmax_kernel_cu(logits, size, stream);
#else
    UNUSED(logits);
    UNUSED(size);
    UNUSED(stream);
    LOG(FATAL) << "CUDA sampling requested in a CPU-only build.";
    return 0;
#endif
}

}  // namespace

size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) {
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        size_t next = std::distance(logits, std::max_element(logits, logits + size));
        return next;
    } else {
        return sample_cuda_argmax(logits, size, stream);
    }
}
}  // namespace sampler
