#include "base/topk_sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace sampler {
TopKSampler::TopKSampler(base::DeviceType device_type, size_t top_k, float temperature,
                         uint32_t seed)
    : Sampler(device_type),
      top_k_(top_k),
      temperature_(temperature),
      rng_(seed == 0 ? std::random_device{}() : seed) {}

size_t TopKSampler::sample(const float* logits, size_t size, void* stream) {
    UNUSED(stream);
    CHECK_NE(logits, nullptr) << "The logits pointer is null.";
    CHECK_GT(size, 0U) << "The logits size must be positive.";
    CHECK_GT(top_k_, 0U) << "Top-k sampling requires top_k > 0.";
    CHECK_GT(temperature_, 0.f) << "Top-k sampling requires temperature > 0.";

    const size_t effective_top_k = std::min(top_k_, size);
    std::vector<size_t> candidate_indices(size);
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0U);

    const auto compare_logits = [logits](size_t lhs, size_t rhs) {
        if (logits[lhs] == logits[rhs]) {
            return lhs < rhs;
        }
        return logits[lhs] > logits[rhs];
    };

    if (effective_top_k < candidate_indices.size()) {
        std::nth_element(candidate_indices.begin(), candidate_indices.begin() + effective_top_k,
                         candidate_indices.end(), compare_logits);
        candidate_indices.resize(effective_top_k);
    }
    std::sort(candidate_indices.begin(), candidate_indices.end(), compare_logits);

    const float max_logit = logits[candidate_indices.front()];
    std::vector<double> weights(candidate_indices.size(), 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < candidate_indices.size(); ++i) {
        const size_t token_idx = candidate_indices.at(i);
        const double shifted_logit =
            static_cast<double>(logits[token_idx] - max_logit) / static_cast<double>(temperature_);
        const double weight = std::exp(shifted_logit);
        weights.at(i) = weight;
        sum += weight;
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        return candidate_indices.front();
    }

    std::uniform_real_distribution<double> distribution(0.0, sum);
    double draw = distribution(rng_);
    for (size_t i = 0; i < candidate_indices.size(); ++i) {
        draw -= weights.at(i);
        if (draw <= 0.0) {
            return candidate_indices.at(i);
        }
    }
    return candidate_indices.back();
}
}  // namespace sampler
