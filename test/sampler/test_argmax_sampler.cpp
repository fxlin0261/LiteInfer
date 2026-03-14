#include <gtest/gtest.h>

#include "sampler/argmax_sampler.h"

TEST(test_argmax_sampler, sample_returns_index_of_largest_logit_on_cpu) {
  sampler::ArgmaxSampler sampler(base::DeviceType::kDeviceCPU);
  const float logits[] = {-1.f, 0.5f, 3.25f, 1.5f};

  EXPECT_EQ(sampler.sample(logits, 4), 2U);
}

TEST(test_argmax_sampler, sample_prefers_first_index_when_logits_tie) {
  sampler::ArgmaxSampler sampler(base::DeviceType::kDeviceCPU);
  const float logits[] = {5.f, 7.f, 7.f, 1.f};

  EXPECT_EQ(sampler.sample(logits, 4), 1U);
}
