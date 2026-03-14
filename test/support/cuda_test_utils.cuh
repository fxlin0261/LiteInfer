#ifndef TEST_SUPPORT_CUDA_TEST_UTILS_CUH_
#define TEST_SUPPORT_CUDA_TEST_UTILS_CUH_

#include <cstdint>

void test_function(float* arr, int32_t size, float value = 1.f);

void set_value_cu(float* arr_cu, int32_t size, float value = 1.f);
#endif  // TEST_SUPPORT_CUDA_TEST_UTILS_CUH_
