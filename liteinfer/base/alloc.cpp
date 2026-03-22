#include "base/alloc.h"
#include <cstring>
#include "base/cuda_config.h"

namespace base {
#if LITEINFER_ENABLE_CUDA
namespace {
void CheckCudaStatus(cudaError_t status, const char* op_name) {
    CHECK(status == cudaSuccess) << op_name << " failed: " << cudaGetErrorString(status);
}
}  // namespace
#endif

void DeviceAllocator::memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                             MemcpyKind memcpy_kind, void* stream, bool need_sync) const {
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if (!byte_size) {
        return;
    }

    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
        return;
    }

#if LITEINFER_ENABLE_CUDA
    cudaStream_t stream_ = nullptr;
    if (stream) {
        stream_ = static_cast<cudaStream_t>(stream);
    }

    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        if (!stream_) {
            CheckCudaStatus(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice),
                            "cudaMemcpy HostToDevice");
        } else {
            CheckCudaStatus(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_),
                "cudaMemcpyAsync HostToDevice");
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        if (!stream_) {
            CheckCudaStatus(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost),
                            "cudaMemcpy DeviceToHost");
        } else {
            CheckCudaStatus(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_),
                "cudaMemcpyAsync DeviceToHost");
        }
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        if (!stream_) {
            CheckCudaStatus(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice),
                            "cudaMemcpy DeviceToDevice");
        } else {
            CheckCudaStatus(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_),
                "cudaMemcpyAsync DeviceToDevice");
        }
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
    }
    if (need_sync) {
        CheckCudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
#else
    LOG(FATAL) << "CUDA memcpy requested in a CPU-only build.";
#endif
}

void DeviceAllocator::memset_zero(void* ptr, size_t byte_size, void* stream, bool need_sync) {
    CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } else {
#if LITEINFER_ENABLE_CUDA
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            CheckCudaStatus(cudaMemsetAsync(ptr, 0, byte_size, stream_), "cudaMemsetAsync");
        } else {
            CheckCudaStatus(cudaMemset(ptr, 0, byte_size), "cudaMemset");
        }
        if (need_sync) {
            CheckCudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
#else
        LOG(FATAL) << "CUDA memset requested in a CPU-only build.";
#endif
    }
}
}  // namespace base
