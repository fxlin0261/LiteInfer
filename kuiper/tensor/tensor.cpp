#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>

namespace tensor {
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
    if (begin >= end) {
        return 0;
    }
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());
    return size;
}

static size_t data_type_size(base::DataType data_type) {
    switch (data_type) {
        case base::DataType::kDataTypeFp32: {
            return 4;
        }
        case base::DataType::kDataTypeInt8: {
            return 1;
        }
        case base::DataType::kDataTypeInt32: {
            return 4;
        }
        default: {
            LOG(FATAL) << "Unknown data type size for " << int(data_type);
            return 0;
        }
    }
}

static std::shared_ptr<base::DeviceAllocator> allocator_from_device_type(
    base::DeviceType device_type) {
    if (device_type == base::DeviceType::kDeviceCPU) {
        return base::CPUDeviceAllocatorFactory::get_instance();
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        return base::CUDADeviceAllocatorFactory::get_instance();
    }
    return nullptr;
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    size_ = dim0;
    init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
    init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
    init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
               bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : data_type_(data_type) {
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;
    init_buffer(alloc, data_type_, need_alloc, ptr);
}

Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims, bool need_alloc,
               std::shared_ptr<base::DeviceAllocator> alloc, void* ptr)
    : dims_(std::move(dims)), data_type_(data_type) {
    size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
    init_buffer(alloc, data_type_, need_alloc, ptr);
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
    } else if (device_type == base::DeviceType::kDeviceCPU) {
        size_t byte_size = this->byte_size();
        auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
        auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
        cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size,
                         base::MemcpyKind::kMemcpyCPU2CUDA, stream);
        this->buffer_ = cu_buffer;
    } else {
        LOG(INFO) << "The device type of the tensor is already cuda.";
    }
}

void Tensor::to_cpu() {
    CHECK_NE(buffer_, nullptr);
    const base::DeviceType device_type = this->device_type();

    if (device_type == base::DeviceType::kDeviceUnknown) {
        LOG(ERROR) << "The device type of the tensor is unknown.";
    } else if (device_type == base::DeviceType::kDeviceCUDA) {
        size_t byte_size = this->byte_size();
        auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
        auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
        cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                          base::MemcpyKind::kMemcpyCUDA2CPU);
        this->buffer_ = cpu_buffer;
    } else {
        LOG(INFO) << "The device type of the tensor is already cpu.";
    }
}

size_t Tensor::size() const { return this->size_; }
int32_t Tensor::get_dim(int32_t idx) const {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_.size());
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
    if (buffer_) {
        if (buffer_->device_type() != base::DeviceType::kDeviceUnknown &&
            buffer->device_type() != base::DeviceType::kDeviceUnknown &&
            buffer_->device_type() != buffer->device_type()) {
            LOG(ERROR) << "The device type of the new buffer is different from the original one.";
            return false;
        }
    }

    size_t byte_size = this->byte_size();
    if (byte_size > buffer->byte_size()) {
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

    size_t byte_size = this->byte_size();
    if (!byte_size) {
        LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
        return false;
    }

    if (buffer_ && byte_size <= buffer_->byte_size()) {
        if (!need_realloc) {
            return true;
        }
    }

    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
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
    this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
    this->buffer_ = nullptr;
}

int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }
base::DataType Tensor::data_type() const { return data_type_; }
void Tensor::reshape(const std::vector<int32_t>& dims) {
    size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
    if (!buffer_) {
        this->dims_ = dims;
        this->size_ = size;
        return;
    }

    if (size > size_) {
        auto allocator = buffer_->allocator();
        if (!allocator) {
            allocator = allocator_from_device_type(buffer_->device_type());
        }
        CHECK(allocator != nullptr) << "Cannot grow a tensor view without a valid device type.";

        auto new_buffer =
            std::make_shared<base::Buffer>(size * base::DataTypeSize(this->data_type_), allocator);
        CHECK(new_buffer->ptr() != nullptr);
        new_buffer->copy_from(buffer_.get());
        this->buffer_ = new_buffer;
    }
    this->dims_ = dims;
    this->size_ = size;
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }
Tensor Tensor::clone() const {
    CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
        << "Cannot clone an empty tensor or a tensor without backing memory.";
    Tensor new_tensor = *this;
    size_t byte_size = this->byte_size();
    auto allocator = buffer_->allocator();
    if (!allocator) {
        allocator = allocator_from_device_type(buffer_->device_type());
    }
    CHECK(allocator != nullptr) << "Cannot clone a tensor view with unknown device type.";
    new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    CHECK(new_tensor.buffer_->ptr() != nullptr);
    new_tensor.buffer_->copy_from(buffer_.get());
    return new_tensor;
}

size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }
std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> strides;
    if (!dims_.empty()) {
        for (int32_t i = 0; i < dims_.size() - 1; ++i) {
            size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
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
        std::shared_ptr<base::Buffer> buffer =
            std::make_shared<base::Buffer>(data_type_size(data_type) * size_, alloc, ptr, true,
                                           alloc ? alloc->device_type() : device_type);
        this->buffer_ = buffer;
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

    if (!allocate(alloc, true)) {
        LOG(ERROR) << "The allocate function failed in init_buffer.";
        this->buffer_ = nullptr;
    }
}
}  // namespace tensor
