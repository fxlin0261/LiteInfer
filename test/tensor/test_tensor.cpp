#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tensor/tensor.h>
#include "base/buffer.h"
#include "support/cuda_test_utils.cuh"

namespace {
bool cuda_available() {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}
}  // namespace

// 看 CUDA tensor 搬回 CPU 后，数据有没有丢
TEST(test_tensor, to_cpu) {
    using namespace base;
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1_cu.is_empty(), false);
    set_value_cu(t1_cu.ptr<float>(), 32 * 32);

    t1_cu.to_cpu();
    ASSERT_EQ(t1_cu.device_type(), base::DeviceType::kDeviceCPU);
    float* cpu_ptr = t1_cu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(*(cpu_ptr + i), 1.f);
    }
}

// 看 clone 出来的 CUDA tensor 里，内容是不是完整拷过去了
TEST(test_tensor, clone_cuda) {
    using namespace base;
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cu(DataType::kDataTypeFp32, 32, 32, true, alloc_cu);
    ASSERT_EQ(t1_cu.is_empty(), false);
    set_value_cu(t1_cu.ptr<float>(), 32 * 32, 1.f);

    tensor::Tensor t2_cu = t1_cu.clone();
    float* p2 = new float[32 * 32];
    cudaMemcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }

    cudaMemcpy(p2, t1_cu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }

    ASSERT_EQ(t2_cu.data_type(), base::DataType::kDataTypeFp32);
    ASSERT_EQ(t2_cu.size(), 32 * 32);

    t2_cu.to_cpu();
    std::memcpy(p2, t2_cu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }
    delete[] p2;
}

// 看 CPU tensor clone 之后，值是不是都还对
TEST(test_tensor, clone_cpu) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    for (int i = 0; i < 32 * 32; ++i) {
        t1_cpu.index<float>(i) = 1.f;
    }

    tensor::Tensor t2_cpu = t1_cpu.clone();
    float* p2 = new float[32 * 32];
    std::memcpy(p2, t2_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }

    std::memcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(p2[i], 1.f);
    }
    delete[] p2;
}

// 看 CPU tensor 搬到 CUDA 后，数据是不是还在
TEST(test_tensor, to_cu) {
    using namespace base;
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA device is not available.";
    }
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    float* p1 = t1_cpu.ptr<float>();
    for (int i = 0; i < 32 * 32; ++i) {
        *(p1 + i) = 1.f;
    }

    t1_cpu.to_cuda();
    float* p2 = new float[32 * 32];
    cudaMemcpy(p2, t1_cpu.ptr<float>(), sizeof(float) * 32 * 32, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32 * 32; ++i) {
        ASSERT_EQ(*(p2 + i), 1.f);
    }
    delete[] p2;
}

// 看按一维大小直接初始化时，buffer 会不会正常分出来
TEST(test_tensor, init1) {
    using namespace base;
    auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, true, alloc_cu);
    ASSERT_EQ(t1.is_empty(), false);
}

// 看用外部指针包一层 tensor 时，是不是直接接住那块内存。
TEST(test_tensor, init3) {
    using namespace base;
    float* ptr = new float[32];
    ptr[0] = 31;
    tensor::Tensor t1(base::DataType::kDataTypeFp32, 32, false, nullptr, ptr);
    ASSERT_EQ(t1.is_empty(), false);
    ASSERT_EQ(t1.ptr<float>(), ptr);
    ASSERT_EQ(*t1.ptr<float>(), 31);
}

// 看不申请内存时，tensor 会不会保持空状态。
TEST(test_tensor, init2) {
    using namespace base;
    auto alloc_cu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    tensor::Tensor t1(base::DataType::kDataTypeFp32, size, false, alloc_cu);
    ASSERT_EQ(t1.is_empty(), true);
}

// 看把外部 buffer 塞进 tensor 后，tensor 能不能正常接管
TEST(test_tensor, assign1) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t1_cpu(DataType::kDataTypeFp32, 32, 32, true, alloc_cpu);
    ASSERT_EQ(t1_cpu.is_empty(), false);

    int32_t size = 32 * 32;
    float* ptr = new float[size];
    for (int i = 0; i < size; ++i) {
        ptr[i] = float(i);
    }
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(size * sizeof(float), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::kDeviceCPU);

    ASSERT_EQ(t1_cpu.assign(buffer), true);
    ASSERT_EQ(t1_cpu.is_empty(), false);
    ASSERT_NE(t1_cpu.ptr<float>(), nullptr);
    delete[] ptr;
}

// 看 clone 是不是真拷贝，不是两个 tensor 共用一块内存
TEST(test_tensor, clone_cpu_is_deep_copy) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor src(DataType::kDataTypeFp32, 8, true, alloc_cpu);
    for (int i = 0; i < 8; ++i) {
        src.index<float>(i) = static_cast<float>(i);
    }

    tensor::Tensor cloned = src.clone();
    ASSERT_NE(cloned.ptr<float>(), src.ptr<float>());

    src.index<float>(0) = 99.f;
    cloned.index<float>(1) = -3.f;

    ASSERT_EQ(src.index<float>(0), 99.f);
    ASSERT_EQ(src.index<float>(1), 1.f);
    ASSERT_EQ(cloned.index<float>(0), 0.f);
    ASSERT_EQ(cloned.index<float>(1), -3.f);
}

// 看 reshape 变大时，前面原有的数据还在不在
TEST(test_tensor, reshape_expand_preserves_existing_data) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 2, 2, true, alloc_cpu);
    for (int i = 0; i < 4; ++i) {
        t.index<float>(i) = static_cast<float>(i + 1);
    }

    t.reshape({2, 3});

    ASSERT_EQ(t.size(), 6);
    ASSERT_EQ(t.dims_size(), 2);
    ASSERT_EQ(t.get_dim(0), 2);
    ASSERT_EQ(t.get_dim(1), 3);
    ASSERT_EQ(t.index<float>(0), 1.f);
    ASSERT_EQ(t.index<float>(1), 2.f);
    ASSERT_EQ(t.index<float>(2), 3.f);
    ASSERT_EQ(t.index<float>(3), 4.f);
}

// 看没分配 buffer 的 tensor 改形状时，只改元信息不乱来。
TEST(test_tensor, reshape_without_buffer_updates_shape_only) {
    tensor::Tensor t(base::DataType::kDataTypeFp32, 2, false, nullptr);
    ASSERT_TRUE(t.is_empty());
    ASSERT_EQ(t.device_type(), base::DeviceType::kDeviceUnknown);

    t.reshape({2, 3, 4});

    ASSERT_EQ(t.size(), 24);
    ASSERT_EQ(t.dims_size(), 3);
    ASSERT_EQ(t.get_dim(0), 2);
    ASSERT_EQ(t.get_dim(1), 3);
    ASSERT_EQ(t.get_dim(2), 4);
    ASSERT_TRUE(t.is_empty());
}

// 看 allocate 该复用时复用，该重分配时重分配。
TEST(test_tensor, allocate_reuses_or_reallocates_buffer) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 16, true, alloc_cpu);

    float* original_ptr = t.ptr<float>();
    ASSERT_NE(original_ptr, nullptr);

    ASSERT_TRUE(t.allocate(alloc_cpu, false));
    ASSERT_EQ(t.ptr<float>(), original_ptr);

    ASSERT_TRUE(t.allocate(alloc_cpu, true));
    ASSERT_NE(t.ptr<float>(), nullptr);
    ASSERT_NE(t.ptr<float>(), original_ptr);
}

// 看 allocate 在没 allocator 或大小是 0 时，会不会老老实实失败。
TEST(test_tensor, allocate_fails_without_allocator_or_size) {
    using namespace base;
    tensor::Tensor t(DataType::kDataTypeFp32, 4, false, nullptr);
    ASSERT_FALSE(t.allocate(nullptr));

    tensor::Tensor empty(DataType::kDataTypeFp32, 0, false, nullptr);
    ASSERT_FALSE(empty.allocate(CPUDeviceAllocatorFactory::get_instance()));
}

// 看 assign 遇到空 buffer 或太小的 buffer，会不会直接拒绝。
TEST(test_tensor, assign_fails_for_null_or_small_buffer) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 8, true, alloc_cpu);

    ASSERT_FALSE(t.assign(nullptr));

    auto small_buffer = std::make_shared<Buffer>(4 * sizeof(float), alloc_cpu);
    ASSERT_FALSE(t.assign(small_buffer));
}

// 看大小合适的外部 buffer 赋进去后，tensor 读出来是不是对的。
TEST(test_tensor, assign_accepts_same_size_external_buffer) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 8, true, alloc_cpu);

    float* ptr = new float[8];
    for (int i = 0; i < 8; ++i) {
        ptr[i] = static_cast<float>(i * 2);
    }

    auto buffer = std::make_shared<Buffer>(8 * sizeof(float), nullptr, ptr, true);
    buffer->set_device_type(DeviceType::kDeviceCPU);

    ASSERT_TRUE(t.assign(buffer));
    for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(t.index<float>(i), static_cast<float>(i * 2));
    }

    delete[] ptr;
}

// 看 shape、stride、byte size 这些基础信息算得对不对。
TEST(test_tensor, metadata_accessors_match_shape_and_type) {
    tensor::Tensor t(base::DataType::kDataTypeInt32, std::vector<int32_t>{2, 3, 4}, false, nullptr);

    ASSERT_EQ(t.data_type(), base::DataType::kDataTypeInt32);
    ASSERT_EQ(t.size(), 24);
    ASSERT_EQ(t.byte_size(), 24 * sizeof(int32_t));
    ASSERT_EQ(t.dims_size(), 3);
    ASSERT_EQ(t.get_dim(0), 2);
    ASSERT_EQ(t.get_dim(1), 3);
    ASSERT_EQ(t.get_dim(2), 4);

    const std::vector<int32_t>& dims = t.dims();
    ASSERT_EQ(dims.size(), 3);
    ASSERT_EQ(dims[0], 2);
    ASSERT_EQ(dims[1], 3);
    ASSERT_EQ(dims[2], 4);

    const std::vector<size_t> strides = t.strides();
    ASSERT_EQ(strides.size(), 3);
    ASSERT_EQ(strides[0], 12);
    ASSERT_EQ(strides[1], 4);
    ASSERT_EQ(strides[2], 1);
}

// 看 reset 之后，旧 buffer 会不会清掉，新形状和类型会不会更新。
TEST(test_tensor, reset_clears_buffer_and_updates_metadata) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 2, 3, true, alloc_cpu);
    ASSERT_FALSE(t.is_empty());
    ASSERT_EQ(t.device_type(), DeviceType::kDeviceCPU);

    t.reset(DataType::kDataTypeInt8, {4, 5});

    ASSERT_TRUE(t.is_empty());
    ASSERT_EQ(t.device_type(), DeviceType::kDeviceUnknown);
    ASSERT_EQ(t.data_type(), DataType::kDataTypeInt8);
    ASSERT_EQ(t.size(), 20);
    ASSERT_EQ(t.byte_size(), 20);
    ASSERT_EQ(t.dims_size(), 2);
    ASSERT_EQ(t.get_dim(0), 4);
    ASSERT_EQ(t.get_dim(1), 5);
    ASSERT_EQ(t.ptr<int8_t>(), nullptr);
}

// 看带偏移的 ptr 能不能拿到对的位置。
TEST(test_tensor, ptr_with_offset_returns_expected_position) {
    using namespace base;
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    tensor::Tensor t(DataType::kDataTypeFp32, 6, true, alloc_cpu);
    for (int i = 0; i < 6; ++i) {
        t.index<float>(i) = static_cast<float>(i + 10);
    }

    float* offset_ptr = t.ptr<float>(3);
    ASSERT_EQ(*offset_ptr, 13.f);

    const tensor::Tensor& const_t = t;
    const float* const_offset_ptr = const_t.ptr<float>(4);
    ASSERT_EQ(*const_offset_ptr, 14.f);
}
