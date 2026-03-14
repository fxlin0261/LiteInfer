#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include "support/cuda_test_utils.cuh"

namespace {

bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

}  // namespace

// 测默认参数时，主机数组会不会都被写成 1.f
TEST(test_cu, test_function_sets_default_value) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    constexpr int32_t size = 32;
    float* ptr = new float[size];
    test_function(ptr, size);
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_FLOAT_EQ(ptr[i], 1.f);
    }
    delete[] ptr;
}

// 测传入自定义值时，主机数组会不会都被写成这个值
TEST(test_cu, test_function_sets_custom_value) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }
    constexpr int32_t size = 32;
    constexpr float value = 3.5f;
    float* ptr = new float[size];
    test_function(ptr, size, value);
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_FLOAT_EQ(ptr[i], value);
    }
    delete[] ptr;
}

// 测只有 1 个元素时，CUDA 包装函数能不能正常写值
TEST(test_cu, test_function_handles_single_element) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    constexpr int32_t size = 1;
    float value = 0.f;
    test_function(&value, size, 2.f);
    ASSERT_FLOAT_EQ(value, 2.f);
}

// 测传空指针时，函数会不会直接返回且不崩溃
TEST(test_cu, test_function_accepts_nullptr) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }
    ASSERT_NO_FATAL_FAILURE(test_function(nullptr, 32));
}

// 测直接写 GPU 内存时，多 block 情况下能不能全部写对
TEST(test_cu, set_value_cu_sets_custom_value_on_device_buffer) {
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }

    constexpr int32_t size = 1025;
    constexpr float value = -2.f;
    float* ptr_cu = nullptr;
    ASSERT_EQ(cudaMalloc(&ptr_cu, sizeof(float) * size), cudaSuccess);

    set_value_cu(ptr_cu, size, value);

    float* host_ptr = new float[size];
    ASSERT_EQ(cudaMemcpy(host_ptr, ptr_cu, sizeof(float) * size, cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int32_t i = 0; i < size; ++i) {
        ASSERT_FLOAT_EQ(host_ptr[i], value);
    }

    delete[] host_ptr;
    ASSERT_EQ(cudaFree(ptr_cu), cudaSuccess);
}
