#include <glog/logging.h>
#include <cstdlib>
#include "base/alloc.h"

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define LITEINFER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {}

void* CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (!byte_size) {
        return nullptr;
    }
#ifdef LITEINFER_HAVE_POSIX_MEMALIGN
    void* data = nullptr;
    const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
    const size_t effective_alignment =
        (alignment >= sizeof(void*)) ? alignment : sizeof(void*);
    const int status = posix_memalign(&data, effective_alignment, byte_size);
    if (status != 0) {
        return nullptr;
    }
    return data;
#else
    void* data = std::malloc(byte_size);
    return data;
#endif
}

void CPUDeviceAllocator::release(void* ptr) const {
    if (ptr) {
        std::free(ptr);
    }
}
}  // namespace base
