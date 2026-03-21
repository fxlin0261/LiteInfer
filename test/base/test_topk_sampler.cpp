#include <gtest/gtest.h>
#include "base/topk_sampler.h"

TEST(test_topk_sampler, top_k_one_matches_greedy_choice_on_cpu) {
    sampler::TopKSampler sampler(base::DeviceType::kDeviceCPU, 1, 0.8f, 123);
    const float logits[] = {-1.f, 0.5f, 3.25f, 1.5f};
    EXPECT_EQ(sampler.sample(logits, 4), 2U);
}

TEST(test_topk_sampler, sampling_stays_within_top_k_candidates) {
    sampler::TopKSampler sampler(base::DeviceType::kDeviceCPU, 2, 1.0f, 123);
    const float logits[] = {10.f, 9.f, 1.f, 0.f};
    for (int i = 0; i < 128; ++i) {
        const size_t sampled = sampler.sample(logits, 4);
        EXPECT_TRUE(sampled == 0U || sampled == 1U);
    }
}
