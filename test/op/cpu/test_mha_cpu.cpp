#include <gtest/gtest.h>
#include <vector>
#include "op/mha.h"

namespace {

tensor::Tensor make_cpu_tensor(base::DataType data_type, const std::vector<int32_t>& dims,
                               void* ptr) {
    tensor::Tensor tensor(data_type, dims, false, nullptr, ptr);
    tensor.set_device_type(base::DeviceType::kDeviceCPU);
    return tensor;
}

}  // namespace

TEST(test_mha_cpu, forward_with_single_position_returns_cached_value_vector) {
    op::MultiHeadAttention layer(base::DeviceType::kDeviceCPU, 0, 1, 2, 1, 1, 2);
    layer.set_pos(0);

    std::vector<float> query_data{1.f, 2.f};
    std::vector<float> score_data{0.f};
    std::vector<float> key_cache_data{3.f, 4.f};
    std::vector<float> value_cache_data{5.f, 6.f};
    std::vector<float> output_data(2, 0.f);

    auto query = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, query_data.data());
    auto score = make_cpu_tensor(base::DataType::kDataTypeFp32, {1}, score_data.data());
    auto key_cache = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, key_cache_data.data());
    auto value_cache = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, value_cache_data.data());
    auto output = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, output_data.data());

    ASSERT_TRUE(layer.forward(query, score, key_cache, value_cache, output));
    EXPECT_FLOAT_EQ(score.index<float>(0), 1.f);
    EXPECT_FLOAT_EQ(output.index<float>(0), 5.f);
    EXPECT_FLOAT_EQ(output.index<float>(1), 6.f);
}

TEST(test_mha_cpu, forward_fails_when_output_tensor_is_empty) {
    op::MultiHeadAttention layer(base::DeviceType::kDeviceCPU, 0, 1, 2, 1, 1, 2);
    layer.set_pos(0);

    std::vector<float> query_data{1.f, 2.f};
    std::vector<float> score_data{0.f};
    std::vector<float> key_cache_data{3.f, 4.f};
    std::vector<float> value_cache_data{5.f, 6.f};

    auto query = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, query_data.data());
    auto score = make_cpu_tensor(base::DataType::kDataTypeFp32, {1}, score_data.data());
    auto key_cache = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, key_cache_data.data());
    auto value_cache = make_cpu_tensor(base::DataType::kDataTypeFp32, {2}, value_cache_data.data());
    tensor::Tensor empty_output;

    const auto status = layer.forward(query, score, key_cache, value_cache, empty_output);
    EXPECT_FALSE(status);
    EXPECT_EQ(status.get_err_code(), base::StatusCode::kInvalidArgument);
}
