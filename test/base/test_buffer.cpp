#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "support/cuda_test_utils.cuh"

namespace {
bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
}  // namespace

// 看最基本的分配是不是正常，顺便确认状态有没有带上
TEST(test_buffer, allocate) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32, alloc);
    ASSERT_NE(buffer.ptr(), nullptr);
    ASSERT_EQ(buffer.allocator(), alloc);
    ASSERT_EQ(buffer.device_type(), DeviceType::kDeviceCPU);
    ASSERT_FALSE(buffer.is_external());
}

// 用外部传进来的内存时，Buffer 应该只是接管指针，不自己重新分配
TEST(test_buffer, use_external) {
    using namespace base;
    float* ptr = new float[32];
    Buffer buffer(32 * sizeof(float), nullptr, ptr, true);
    buffer.set_device_type(DeviceType::kDeviceCPU);

    ASSERT_EQ(buffer.ptr(), ptr);
    ASSERT_EQ(buffer.byte_size(), 32 * sizeof(float));
    ASSERT_TRUE(buffer.is_external());
    ASSERT_EQ(buffer.device_type(), DeviceType::kDeviceCPU);
    delete[] ptr;
}

// 单独测一下 allocate() 这个接口能不能真正分到内存
TEST(test_buffer, allocate_method) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

    Buffer buffer(16 * sizeof(float), alloc, nullptr, true);

    ASSERT_TRUE(buffer.allocate());
    ASSERT_NE(buffer.ptr(), nullptr);
    ASSERT_FALSE(buffer.is_external());
}

// 没给 allocator 的话，allocate() 应该直接失败
TEST(test_buffer, allocate_fails_without_allocator) {
    using namespace base;

    Buffer buffer(16 * sizeof(float), nullptr, nullptr, false);

    ASSERT_FALSE(buffer.allocate());
    ASSERT_EQ(buffer.ptr(), nullptr);
}

// 大小是 0 的时候，没必要分配，应该失败
TEST(test_buffer, allocate_fails_with_zero_size) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

    Buffer buffer(0, alloc);

    ASSERT_FALSE(buffer.allocate());
}

// CPU 拷到 CPU 时，只会按两边里更小的那部分去拷
TEST(test_buffer, cpu_memcpy_copies_minimum_byte_size) {
    using namespace base;
    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();

    Buffer src(4 * sizeof(float), alloc);
    Buffer dst(2 * sizeof(float), alloc);
    src.set_device_type(DeviceType::kDeviceCPU);
    dst.set_device_type(DeviceType::kDeviceCPU);

    auto* src_ptr = static_cast<float*>(src.ptr());
    auto* dst_ptr = static_cast<float*>(dst.ptr());
    src_ptr[0] = 1.f;
    src_ptr[1] = 2.f;
    src_ptr[2] = 3.f;
    src_ptr[3] = 4.f;
    dst_ptr[0] = 0.f;
    dst_ptr[1] = 0.f;

    dst.copy_from(src);

    ASSERT_EQ(dst_ptr[0], 1.f);
    ASSERT_EQ(dst_ptr[1], 2.f);
}

// 外部 CPU 数据拷到 CUDA 上，内容要完整过去
TEST(test_buffer, cuda_memcpy1) {
    using namespace base;
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32;
    float* ptr = new float[size];
    for (int i = 0; i < size; ++i) {
        ptr[i] = float(i);
    }
    Buffer buffer(size * sizeof(float), nullptr, ptr, true);
    buffer.set_device_type(DeviceType::kDeviceCPU);
    ASSERT_TRUE(buffer.is_external());

    Buffer cu_buffer(size * sizeof(float), alloc_cu);
    cu_buffer.copy_from(buffer);

    float* ptr2 = new float[size];
    cudaMemcpy(ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(ptr2[i], float(i));
    }

    delete[] ptr;
    delete[] ptr2;
}

// CUDA 拷到 CUDA，数据别丢也别乱
TEST(test_buffer, cuda_memcpy3) {
    using namespace base;
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32;
    Buffer cu_buffer1(size * sizeof(float), alloc_cu);
    Buffer cu_buffer2(size * sizeof(float), alloc_cu);

    set_value_cu((float*)cu_buffer2.ptr(), size);
    // cu to cu
    ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
    ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCUDA);

    cu_buffer1.copy_from(cu_buffer2);

    float* ptr2 = new float[size];
    cudaMemcpy(ptr2, cu_buffer1.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(ptr2[i], 1.f);
    }
    delete[] ptr2;
}

// CUDA 拷回 CPU，拿到的值应该还是对的
TEST(test_buffer, cuda_memcpy4) {
    using namespace base;
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    auto alloc = base::CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32;
    Buffer cu_buffer1(size * sizeof(float), alloc_cu);
    Buffer cu_buffer2(size * sizeof(float), alloc);
    ASSERT_EQ(cu_buffer1.device_type(), DeviceType::kDeviceCUDA);
    ASSERT_EQ(cu_buffer2.device_type(), DeviceType::kDeviceCPU);

    // cu to cpu
    set_value_cu((float*)cu_buffer1.ptr(), size);
    cu_buffer2.copy_from(cu_buffer1);

    float* ptr2 = (float*)cu_buffer2.ptr();
    for (int i = 0; i < size; ++i) {
        ASSERT_EQ(ptr2[i], 1.f);
    }
}
