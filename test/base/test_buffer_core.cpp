//
// Created by fss on 9/19/24.
//
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "support/cuda_test_utils.cuh"

namespace {

bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

}  // namespace

// 指定大小和分配器分配内存
TEST(test_buffer, constructor_allocates_with_allocator) {
    using namespace base;

    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(32 * sizeof(float), alloc);

    ASSERT_NE(buffer.ptr(), nullptr);
    EXPECT_EQ(buffer.allocator(), alloc);
    EXPECT_EQ(buffer.device_type(), DeviceType::kDeviceCPU);
    EXPECT_FALSE(buffer.is_external());
}

// 外部已分配内存进行绑定
TEST(test_buffer, external_cpu_memory_keeps_original_pointer) {
    using namespace base;

    float* ptr = new float[32];
    Buffer buffer(32 * sizeof(float), nullptr, ptr, true);
    buffer.set_device_type(DeviceType::kDeviceCPU);

    EXPECT_EQ(buffer.ptr(), ptr);
    EXPECT_EQ(buffer.byte_size(), 32 * sizeof(float));
    EXPECT_TRUE(buffer.is_external());
    EXPECT_EQ(buffer.device_type(), DeviceType::kDeviceCPU);

    delete[] ptr;
}

// 没给外部内存，但设置成 true
TEST(test_buffer, allocate_succeeds_when_allocator_and_size_are_valid) {
    using namespace base;

    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(16 * sizeof(float), alloc, nullptr, true);

    ASSERT_TRUE(buffer.allocate());
    EXPECT_NE(buffer.ptr(), nullptr);
    EXPECT_FALSE(buffer.is_external());
}

// 没有 allocator 时分配失败
TEST(test_buffer, allocate_fails_without_allocator) {
    using namespace base;

    Buffer buffer(16 * sizeof(float), nullptr, nullptr, false);

    EXPECT_FALSE(buffer.allocate());
    EXPECT_EQ(buffer.ptr(), nullptr);
}

// 申请 0 字节时分配失败
TEST(test_buffer, allocate_fails_with_zero_size) {
    using namespace base;

    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    Buffer buffer(0, alloc);

    EXPECT_FALSE(buffer.allocate());
}

// 测 CPU 到 CPU 拷贝，只拷最小字节数
TEST(test_buffer, copy_from_cpu_to_cpu_copies_minimum_byte_size) {
    using namespace base;

    auto alloc = CPUDeviceAllocatorFactory::get_instance();
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

    EXPECT_FLOAT_EQ(dst_ptr[0], 1.f);
    EXPECT_FLOAT_EQ(dst_ptr[1], 2.f);
}

// 测外部 CPU Buffer 拷到 CUDA Buffer
TEST(test_buffer, copy_from_cpu_to_cuda_works_with_external_source) {
    using namespace base;

    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    constexpr int32_t size = 32;

    float* src_ptr = new float[size];
    for (int32_t i = 0; i < size; ++i) {
        src_ptr[i] = static_cast<float>(i);
    }

    Buffer src(size * sizeof(float), nullptr, src_ptr, true);
    src.set_device_type(DeviceType::kDeviceCPU);

    Buffer dst(size * sizeof(float), alloc_cu);
    ASSERT_EQ(dst.device_type(), DeviceType::kDeviceCUDA);

    dst.copy_from(src);

    float host_copy[size] = {0.f};
    ASSERT_EQ(cudaMemcpy(host_copy, dst.ptr(), sizeof(host_copy), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int32_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(host_copy[i], static_cast<float>(i));
    }

    delete[] src_ptr;
}

// 测 CUDA 到 CPU 拷贝
TEST(test_buffer, copy_from_cuda_to_cpu_works) {
    using namespace base;

    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    auto alloc = CPUDeviceAllocatorFactory::get_instance();
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    constexpr int32_t size = 32;

    Buffer src(size * sizeof(float), alloc_cu);
    Buffer dst(size * sizeof(float), alloc);

    ASSERT_EQ(src.device_type(), DeviceType::kDeviceCUDA);
    ASSERT_EQ(dst.device_type(), DeviceType::kDeviceCPU);

    set_value_cu(static_cast<float*>(src.ptr()), size);
    dst.copy_from(src);

    auto* dst_ptr = static_cast<float*>(dst.ptr());
    for (int32_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(dst_ptr[i], 1.f);
    }
}

// 测 CUDA 到 CUDA 拷贝
TEST(test_buffer, copy_from_cuda_to_cuda_works) {
    using namespace base;

    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    constexpr int32_t size = 32;

    Buffer src(size * sizeof(float), alloc_cu);
    Buffer dst(size * sizeof(float), alloc_cu);

    ASSERT_EQ(src.device_type(), DeviceType::kDeviceCUDA);
    ASSERT_EQ(dst.device_type(), DeviceType::kDeviceCUDA);

    set_value_cu(static_cast<float*>(src.ptr()), size);
    dst.copy_from(src);

    float host_copy[size] = {0.f};
    ASSERT_EQ(cudaMemcpy(host_copy, dst.ptr(), sizeof(host_copy), cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int32_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(host_copy[i], 1.f);
    }
}
