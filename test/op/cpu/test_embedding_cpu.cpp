#include <gtest/gtest.h>

#include <vector>

#include "op/embedding.h"

namespace {

tensor::Tensor make_cpu_tensor(base::DataType data_type, const std::vector<int32_t>& dims, void* ptr) {
  tensor::Tensor tensor(data_type, dims, false, nullptr, ptr);
  tensor.set_device_type(base::DeviceType::kDeviceCPU);
  return tensor;
}

}  // namespace

TEST(test_embedding_cpu, forward_copies_selected_rows_from_weight_matrix) {
  op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, 3, 8, 4);

  std::vector<int32_t> tokens{2, 0, 3};
  std::vector<int32_t> token_count_placeholder{0, 0, 0};
  std::vector<float> weight{
      1.f,  2.f,  3.f,   //
      4.f,  5.f,  6.f,   //
      7.f,  8.f,  9.f,   //
      10.f, 11.f, 12.f,
  };
  std::vector<float> output_data(9, 0.f);

  auto input_tokens = make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, tokens.data());
  auto input_token_num =
      make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_count_placeholder.data());
  auto output = make_cpu_tensor(base::DataType::kDataTypeFp32, {3, 3}, output_data.data());
  ASSERT_TRUE(layer.set_weight(0, {4, 3}, weight.data(), base::DeviceType::kDeviceCPU));

  ASSERT_TRUE(layer.forward(input_tokens, input_token_num, output));
  EXPECT_FLOAT_EQ(output.index<float>(0), 7.f);
  EXPECT_FLOAT_EQ(output.index<float>(1), 8.f);
  EXPECT_FLOAT_EQ(output.index<float>(2), 9.f);
  EXPECT_FLOAT_EQ(output.index<float>(3), 1.f);
  EXPECT_FLOAT_EQ(output.index<float>(4), 2.f);
  EXPECT_FLOAT_EQ(output.index<float>(5), 3.f);
  EXPECT_FLOAT_EQ(output.index<float>(6), 10.f);
  EXPECT_FLOAT_EQ(output.index<float>(7), 11.f);
  EXPECT_FLOAT_EQ(output.index<float>(8), 12.f);
}

TEST(test_embedding_cpu, forward_fails_when_output_shape_does_not_match_token_count) {
  op::EmbeddingLayer layer(base::DeviceType::kDeviceCPU, 3, 8, 4);

  std::vector<int32_t> tokens{1, 2, 3};
  std::vector<int32_t> token_count_placeholder{0, 0, 0};
  std::vector<float> weight(12, 1.f);
  std::vector<float> output_data(6, 0.f);

  auto input_tokens = make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, tokens.data());
  auto input_token_num =
      make_cpu_tensor(base::DataType::kDataTypeInt32, {3}, token_count_placeholder.data());
  auto output = make_cpu_tensor(base::DataType::kDataTypeFp32, {2, 3}, output_data.data());
  ASSERT_TRUE(layer.set_weight(0, {4, 3}, weight.data(), base::DeviceType::kDeviceCPU));

  const auto status = layer.forward(input_tokens, input_token_num, output);
  EXPECT_FALSE(status);
  EXPECT_EQ(status.get_err_code(), base::StatusCode::kInvalidArgument);
}
