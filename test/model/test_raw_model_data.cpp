#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "model/raw_model_data.h"

TEST(test_raw_model_data, fp32_weight_interprets_offset_in_elements) {
  std::array<float, 4> weights{1.f, 2.f, 3.f, 4.f};

  model::RawModelDataFp32 model_data;
  model_data.weight_data = weights.data();

  const auto* ptr = static_cast<const float*>(model_data.weight(2));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr, weights.data() + 2);
  EXPECT_FLOAT_EQ(*ptr, 3.f);
}

TEST(test_raw_model_data, int8_weight_interprets_offset_in_bytes) {
  std::array<int8_t, 4> weights{1, 2, 3, 4};

  model::RawModelDataInt8 model_data;
  model_data.weight_data = weights.data();

  const auto* ptr = static_cast<const int8_t*>(model_data.weight(2));
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ptr, weights.data() + 2);
  EXPECT_EQ(*ptr, 3);
}
