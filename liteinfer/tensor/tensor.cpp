#include "tensor/tensor.h"
#include <functional>
#include <glog/logging.h>
#include <numeric>
#include <utility>

namespace tensor {
namespace {

template <typename Iterator, typename Value>
size_t reduce_dimension(Iterator begin, Iterator end, Value init) {
    if (begin >= end) {
        return 0;
    }
    return std::accumulate(begin, end, init, std::multiplies<>());
}

size_t compute_element_count(const std::vector<int32_t>& dims) {
    return reduce_dimension(dims.begin(), dims.end(), size_t{1});
}

size_t checked_byte_size(base::DataType data_type, size_t element_count) {
    const size_t type_size = base::DataTypeSize(data_type);
    CHECK_NE(type_size, 0U) << "Unknown data type size for " << int(data_type);
    return type_size * element_count;
}

std::shared_ptr<base::DeviceAllocator> allocator_from_device_type(base::DeviceType device_type) {
    switch (device_type) {
        case base::DeviceType::kDeviceCPU:
            return base::CPUDeviceAllocatorFactory::get_instance();
        case base::DeviceType::kDeviceCUDA:
            return base::CUDADeviceAllocatorFactory::get_instance();
        case base::DeviceType::kDeviceUnknown:
        default:
            return nullptr;
    }
}

std::shared_ptr<base::DeviceAllocator> resolve_allocator(
    const std::shared_ptr<base::Buffer>& buffer) {
    CHECK_NE(buffer, nullptr);
    auto allocator = buffer->allocator();
    if (allocator) {
        return allocator;
    }
    return allocator_from_device_type(buffer->device_type());
}

bool has_compatible_device_type(const std::shared_ptr<base::Buffer>& current_buffer,
                                const std::shared_ptr<base::Buffer>& new_buffer) {
    CHECK_NE(current_buffer, nullptr);
    CHECK_NE(new_buffer, nullptr);

    const base::DeviceType current_device_type = current_buffer->device_type();
    const base::DeviceType new_device_type = new_buffer->device_type();
    return current_device_type == base::DeviceType::kDeviceUnknown ||
           new_device_type == base::DeviceType::kDeviceUnknown ||
           current_device_type == new_device_type;
}

std::shared_ptr<base::Buffer> copy_buffer_to_device(const std::shared_ptr<base::Buffer>& source,
                                                    size_t byte_size,
                                                    base::DeviceType target_device,
                                                    base::MemcpyKind memcpy_kind,
                                                    cudaStream_t stream = nullptr) {
    CHECK_NE(source, nullptr);
    const auto allocator = allocator_from_device_type(target_device);
    CHECK_NE(allocator, nullptr)
        << "Cannot resolve allocator for target device type " << int(target_device);

    auto destination = std::make_shared<base::Buffer>(byte_size, allocator);
    CHECK_NE(destination->ptr(), nullptr) << "The copied buffer points to a null pointer.";
    allocator->memcpy(source->ptr(), destination->ptr(), byte_size, memcpy_kind, stream);
    return destination;
}

}  // namespace

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : Tensor(data_type, std::vector<int32_t>{dim0}, need_alloc, std::move(alloc), ptr) {}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : Tensor(data_type, std::vector<int32_t>{dim0, dim1}, need_alloc, std::move(alloc), ptr) {}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : Tensor(data_type, std::vector<int32_t>{dim0, dim1, dim2}, need_alloc, std::move(alloc),
             ptr) {}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : Tensor(data_type, std::vector<int32_t>{dim0, dim1, dim2, dim3}, need_alloc,
             std::move(alloc), ptr) {}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : size_(compute_element_count(dims)), dims_(std::move(dims)), data_type_(data_type) {
    init_buffer(std::move(alloc), data_type_, need_alloc, ptr);
}

Tensor Tensor::make_external(base::DataType data_type, std::vector<int32_t> dims, void* ptr,
                             base::DeviceType device_type) {
    CHECK_NE(ptr, nullptr) << "The ptr parameter in make_external is a null pointer.";
    Tensor tensor(data_type, std::move(dims));
    tensor.init_buffer(nullptr, data_type, false, ptr, device_type);
    return tensor;
}

void Tensor::to_cuda(cudaStream_t stream) {
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type = this->device_type();
    if (device_type == base::DeviceType::kDeviceUnknown) {
        LOG(ERROR) << "The device type of the tensor is unknown.";
        return;
    }
    if (device_type == base::DeviceType::kDeviceCUDA) {
        LOG(INFO) << "The device type of the tensor is already cuda.";
        return;
    }

    this->buffer_ = copy_buffer_to_device(buffer_, this->byte_size(), base::DeviceType::kDeviceCUDA,
                                          base::MemcpyKind::kMemcpyCPU2CUDA, stream);
}

void Tensor::to_cpu() {
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type = this->device_type();

    if (device_type == base::DeviceType::kDeviceUnknown) {
        LOG(ERROR) << "The device type of the tensor is unknown.";
        return;
    }
    if (device_type == base::DeviceType::kDeviceCPU) {
        LOG(INFO) << "The device type of the tensor is already cpu.";
        return;
    }

    this->buffer_ = copy_buffer_to_device(buffer_, this->byte_size(), base::DeviceType::kDeviceCPU,
                                          base::MemcpyKind::kMemcpyCUDA2CPU);
}

size_t Tensor::size() const { return this->size_; }

int32_t Tensor::get_dim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_size());
    return this->dims_.at(idx);
}

base::DeviceType Tensor::device_type() const {
    if (!buffer_) {
        return base::DeviceType::kDeviceUnknown;
    }
    return buffer_->device_type();
}

bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
    if (!buffer) {
        LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
        return false;
    }
    if (buffer_ && !has_compatible_device_type(buffer_, buffer)) {
        LOG(ERROR) << "The device type of the new buffer is different from the original one.";
        return false;
    }

    const size_t required_byte_size = this->byte_size();
    if (required_byte_size > buffer->byte_size()) {
        LOG(ERROR) << "The size of buffer is too small for the tensor!";
        return false;
    }
    buffer_ = buffer;
    return true;
}

bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
    if (!allocator) {
        LOG(ERROR) << "The allocator parameter in the allocate function is null "
                      "pointer!";
        return false;
    }

    const size_t required_byte_size = this->byte_size();
    if (!required_byte_size) {
        LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
        return false;
    }

    if (buffer_ && required_byte_size <= buffer_->byte_size() && !need_realloc) {
        return true;
    }

    buffer_ = std::make_shared<base::Buffer>(required_byte_size, allocator);
    if (!buffer_->ptr()) {
        LOG(ERROR) << "The memory allocated is a null pointer!";
        return false;
    }
    return true;
}

const std::vector<int32_t>& Tensor::dims() const { return this->dims_; }

void Tensor::set_device_type(base::DeviceType device_type) const {
    if (buffer_) {
        buffer_->set_device_type(device_type);
    }
}

void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
    this->data_type_ = data_type;
    this->dims_ = dims;
    this->size_ = compute_element_count(dims_);
    this->buffer_ = nullptr;
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

base::DataType Tensor::data_type() const { return data_type_; }

void Tensor::reshape(const std::vector<int32_t>& dims) {
    const size_t new_size = compute_element_count(dims);
    if (!buffer_) {
        this->dims_ = dims;
        this->size_ = new_size;
        return;
    }

    if (new_size > size_) {
        auto allocator = resolve_allocator(buffer_);
        CHECK(allocator != nullptr) << "Cannot grow a tensor view without a valid device type.";

        auto new_buffer =
            std::make_shared<base::Buffer>(checked_byte_size(this->data_type_, new_size), allocator);
        CHECK(new_buffer->ptr() != nullptr);
        new_buffer->copy_from(buffer_.get());
        this->buffer_ = new_buffer;
    }
    this->dims_ = dims;
    this->size_ = new_size;
}

std::shared_ptr<base::Buffer> Tensor::get_runtime_tensor() const { return buffer_; }

Tensor Tensor::clone() const {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "Cannot clone an empty tensor or a tensor without backing memory.";
    Tensor new_tensor = *this;
    const size_t clone_byte_size = this->byte_size();
    auto allocator = resolve_allocator(buffer_);
    CHECK(allocator != nullptr) << "Cannot clone a tensor view with unknown device type.";
    new_tensor.buffer_ = std::make_shared<base::Buffer>(clone_byte_size, allocator);
    CHECK(new_tensor.buffer_->ptr() != nullptr);
    new_tensor.buffer_->copy_from(buffer_.get());
    return new_tensor;
}

size_t Tensor::byte_size() const { return this->size() * base::DataTypeSize(data_type_); }

std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> strides;
    if (!dims_.empty()) {
        strides.reserve(dims_.size());
        for (size_t i = 0; i + 1 < dims_.size(); ++i) {
            size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), size_t{1});
            strides.push_back(stride);
        }
        strides.push_back(1);
    }
    return strides;
}

bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc, base::DataType data_type,
                         bool need_alloc, void* ptr, base::DeviceType device_type) {
    CHECK(!need_alloc || ptr == nullptr)
        << "The need_alloc parameter cannot be true when ptr is not a null pointer.";

    if (ptr != nullptr) {
        this->buffer_ = std::make_shared<base::Buffer>(
            checked_byte_size(data_type, size_), alloc, ptr, true,
            alloc ? alloc->device_type() : device_type);
        return;
    }

    if (!need_alloc) {
        this->buffer_ = nullptr;
        return;
    }

    if (!alloc) {
        LOG(ERROR) << "The allocator parameter in init_buffer is null when allocation is needed.";
        this->buffer_ = nullptr;
        return;
    }

    if (!allocate(std::move(alloc), true)) {
        LOG(ERROR) << "The allocate function failed in init_buffer.";
        this->buffer_ = nullptr;
    }
}
}  // namespace tensor
