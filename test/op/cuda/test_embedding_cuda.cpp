#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "base/buffer.h"
#include "op/kernels/kernels_interface.h"

// 测试默认流下，是否能取到第 1 个 token 的 embedding。
TEST(test_emb_cu, emb1_nostream) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4;    //  词表大小
    int32_t dim = 512;    //  词向量维度
    int32_t size = 2048;  //  总元素数
    // 构造输入 token
    tensor::Tensor input(base::DataType::kDataTypeFp32, 1, true, alloc_cpu);
    // 这里创建了一个长度为 1 的输入张量，里面存的是 token id  1
    input.index<int32_t>(0) = 1;
    // weight 是 CPU 上的 4 x 512 浮点矩阵
    tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    // output 是 GPU 上的长度为 512 的浮点向量
    tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);
    // 这里把 weight 填成连续值：
    for (int i = 0; i < size; ++i) {
        weight.index<float>(i) = static_cast<float>(i);
    }
    // 把权重拷到 CUDA，并执行 embedding
    weight.to_cuda();
    kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token, nullptr);
    // 拷回 CPU 并校验结果
    output.to_cpu();
    // 这里逐个检查输出的 512 个元素是否等于 512 + i
    for (int i = 0; i < dim; ++i) {
        ASSERT_EQ(output.index<float>(i), 512 + i);
    }
}

// 测试默认流下，是否能取到第 2 个 token 的 embedding。
TEST(test_emb_cu, emb2_nostream) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4;
    int32_t dim = 512;
    int32_t size = 2048;

    tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 2;

    tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

    for (int i = 0; i < size; ++i) {
        weight.index<float>(i) = static_cast<float>(i);
    }
    weight.to_cuda();
    kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token, nullptr);

    output.to_cpu();
    for (int i = 0; i < dim; ++i) {
        ASSERT_EQ(output.index<float>(i), 1024 + i);
    }
}

// 测试传入自定义流时，embedding 查询是否也能正确执行。
TEST(test_emb_cu, emb1_stream) {
    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = base::CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4;
    int32_t dim = 512;
    int32_t size = 2048;

    tensor::Tensor input(base::DataType::kDataTypeInt32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 1;

    tensor::Tensor weight(base::DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    tensor::Tensor output(base::DataType::kDataTypeFp32, dim, true, alloc_cu);

    for (int i = 0; i < size; ++i) {
        weight.index<float>(i) = static_cast<float>(i);
    }
    weight.to_cuda();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    kernel::get_emb_kernel(base::DeviceType::kDeviceCUDA)(input, weight, output, token, stream);

    output.to_cpu();
    for (int i = 0; i < dim; ++i) {
        ASSERT_EQ(output.index<float>(i), 512 + i);
    }

    cudaStreamDestroy(stream);
}
