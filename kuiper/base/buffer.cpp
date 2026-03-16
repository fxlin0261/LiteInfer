#include "base/buffer.h"
#include <glog/logging.h>
#include <algorithm>

namespace base {
namespace {
MemcpyKind resolve_memcpy_kind(DeviceType src_device, DeviceType dst_device) {
    CHECK(src_device != DeviceType::kDeviceUnknown);
    CHECK(dst_device != DeviceType::kDeviceUnknown);

    if (src_device == DeviceType::kDeviceCPU && dst_device == DeviceType::kDeviceCPU) {
        return MemcpyKind::kMemcpyCPU2CPU;
    } else if (src_device == DeviceType::kDeviceCUDA && dst_device == DeviceType::kDeviceCPU) {
        return MemcpyKind::kMemcpyCUDA2CPU;
    } else if (src_device == DeviceType::kDeviceCPU && dst_device == DeviceType::kDeviceCUDA) {
        return MemcpyKind::kMemcpyCPU2CUDA;
    } else {
        return MemcpyKind::kMemcpyCUDA2CUDA;
    }
}
}  // namespace

Buffer::Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator, void* ptr,
               bool use_external, DeviceType device_type)
    : byte_size_(byte_size),
      ptr_(ptr),
      use_external_(use_external),
      device_type_(device_type),
      allocator_(std::move(allocator)) {
    if (allocator_) {
        const DeviceType allocator_device_type = allocator_->device_type();
        if (device_type_ != DeviceType::kDeviceUnknown && device_type_ != allocator_device_type) {
            LOG(WARNING) << "The explicit device type does not match the allocator device type. "
                         << "Using the allocator device type instead.";
        }
        device_type_ = allocator_device_type;
    }

    if (!ptr_ && allocator_) {
        use_external_ = false;
        ptr_ = allocator_->allocate(byte_size_);
    }
}

Buffer::~Buffer() {
    if (!use_external_) {
        if (ptr_ && allocator_) {
            allocator_->release(ptr_);
            ptr_ = nullptr;
        }
    }
}

void* Buffer::ptr() { return ptr_; }
const void* Buffer::ptr() const { return ptr_; }
size_t Buffer::byte_size() const { return byte_size_; }
bool Buffer::allocate() {
    if (!allocator_ || byte_size_ == 0) {
        return false;
    }

    if (ptr_ && !use_external_) {
        allocator_->release(ptr_);
        ptr_ = nullptr;
    }

    use_external_ = false;
    device_type_ = allocator_->device_type();
    ptr_ = allocator_->allocate(byte_size_);
    return ptr_ != nullptr;
}

std::shared_ptr<DeviceAllocator> Buffer::allocator() const { return allocator_; }
void Buffer::copy_from(const Buffer& buffer) const {
    CHECK(allocator_ != nullptr);
    CHECK(ptr_ != nullptr);
    CHECK(buffer.ptr_ != nullptr);
    size_t byte_size = std::min(byte_size_, buffer.byte_size_);
    const DeviceType& buffer_device = buffer.device_type();
    const DeviceType& current_device = this->device_type();
    CHECK(buffer_device != DeviceType::kDeviceUnknown &&
          current_device != DeviceType::kDeviceUnknown);

    return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                              resolve_memcpy_kind(buffer_device, current_device));
}

void Buffer::copy_from(const Buffer* buffer) const {
    CHECK_NE(buffer, nullptr);
    copy_from(*buffer);
}
DeviceType Buffer::device_type() const { return device_type_; }
void Buffer::set_device_type(DeviceType device_type) {
    if (allocator_) {
        const DeviceType allocator_device_type = allocator_->device_type();
        if (device_type != DeviceType::kDeviceUnknown && device_type != allocator_device_type) {
            LOG(WARNING) << "Cannot override buffer device type to a different device than the "
                         << "allocator. Keeping the allocator device type instead.";
        }
        device_type_ = allocator_device_type;
        return;
    }
    device_type_ = device_type;
}

std::shared_ptr<Buffer> Buffer::get_shared_from_this() { return shared_from_this(); }
bool Buffer::is_external() const { return this->use_external_; }
}  // namespace base
