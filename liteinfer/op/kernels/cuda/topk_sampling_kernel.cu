#include "topk_sampling_kernel.cuh"

#include <cfloat>
#include <cmath>
#include <cub/cub.cuh>
#include "base/cuda_config.h"
#include "base/alloc.h"

namespace kernel {
namespace {
constexpr size_t kMaxSupportedTopK = 256;
constexpr int kTopKSamplingThreads = 256;

struct TopKCandidate {
    float value = -FLT_MAX;
    size_t index = SIZE_MAX;
};

struct TopKCandidateReduceOp {
    __device__ TopKCandidate operator()(const TopKCandidate& lhs,
                                        const TopKCandidate& rhs) const {
        if (lhs.index == SIZE_MAX) {
            return rhs;
        }
        if (rhs.index == SIZE_MAX) {
            return lhs;
        }
        if (rhs.value > lhs.value) {
            return rhs;
        }
        if (rhs.value == lhs.value && rhs.index < lhs.index) {
            return rhs;
        }
        return lhs;
    }
};

__device__ bool is_already_selected(size_t token_idx, const size_t* selected_indices,
                                    size_t selected_count) {
    for (size_t i = 0; i < selected_count; ++i) {
        if (selected_indices[i] == token_idx) {
            return true;
        }
    }
    return false;
}

__global__ void topk_sampling_kernel_fp32(const float* logits, size_t size, size_t top_k,
                                          float temperature, double random_value,
                                          size_t* output_idx) {
    if (blockIdx.x != 0) {
        return;
    }

    using BlockReduce = cub::BlockReduce<TopKCandidate, kTopKSamplingThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float top_values[kMaxSupportedTopK];
    __shared__ size_t top_indices[kMaxSupportedTopK];
    __shared__ double weights[kMaxSupportedTopK];
    __shared__ size_t candidate_count;

    if (threadIdx.x == 0) {
        candidate_count = top_k < size ? top_k : size;
    }
    __syncthreads();

    for (size_t selected_count = 0; selected_count < candidate_count; ++selected_count) {
        TopKCandidate local_best;
        for (size_t token_idx = threadIdx.x; token_idx < size; token_idx += blockDim.x) {
            if (is_already_selected(token_idx, top_indices, selected_count)) {
                continue;
            }

            const TopKCandidate candidate{logits[token_idx], token_idx};
            local_best = TopKCandidateReduceOp{}(local_best, candidate);
        }

        const TopKCandidate block_best =
            BlockReduce(reduce_storage).Reduce(local_best, TopKCandidateReduceOp{});
        __syncthreads();
        if (threadIdx.x == 0) {
            top_values[selected_count] = block_best.value;
            top_indices[selected_count] = block_best.index;
        }
        __syncthreads();
    }

    if (threadIdx.x != 0) {
        return;
    }

    const float max_logit = top_values[0];
    double sum = 0.0;
    for (size_t i = 0; i < candidate_count; ++i) {
        const double shifted_logit =
            static_cast<double>(top_values[i] - max_logit) / static_cast<double>(temperature);
        weights[i] = exp(shifted_logit);
        sum += weights[i];
    }

    if (!(sum > 0.0) || !isfinite(sum)) {
        *output_idx = top_indices[0];
        return;
    }

    double draw = random_value * sum;
    for (size_t i = 0; i < candidate_count; ++i) {
        draw -= weights[i];
        if (draw <= 0.0) {
            *output_idx = top_indices[i];
            return;
        }
    }
    *output_idx = top_indices[candidate_count - 1];
}
}  // namespace

size_t topk_sampling_kernel_cu(const float* logits, size_t size, size_t top_k, float temperature,
                               double random_value, void* stream) {
    CHECK_NE(logits, nullptr) << "The logits pointer is null.";
    CHECK_GT(size, 0U) << "The logits size must be positive.";
    CHECK_GT(top_k, 0U) << "Top-k sampling requires top_k > 0.";
    CHECK_GT(temperature, 0.f) << "Top-k sampling requires temperature > 0.";
    CHECK_LE(top_k, kMaxSupportedTopK)
        << "The CUDA top-k sampling kernel supports top_k up to " << kMaxSupportedTopK << ".";
    CHECK_GE(random_value, 0.0);
    CHECK_LE(random_value, 1.0);

    auto alloc_cu = base::CUDADeviceAllocatorFactory::get_instance();
    size_t* output_idx = static_cast<size_t*>(alloc_cu->allocate(sizeof(size_t)));
    CHECK_NE(output_idx, nullptr) << "Failed to allocate CUDA memory for top-k sampling output.";

    size_t host_output_idx = 0;
    if (stream == nullptr) {
        topk_sampling_kernel_fp32<<<1, kTopKSamplingThreads>>>(logits, size, top_k, temperature,
                                                               random_value, output_idx);
        cudaMemcpy(&host_output_idx, output_idx, sizeof(size_t), cudaMemcpyDeviceToHost);
    } else {
        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        topk_sampling_kernel_fp32<<<1, kTopKSamplingThreads, 0, cuda_stream>>>(
            logits, size, top_k, temperature, random_value, output_idx);
        cudaMemcpyAsync(&host_output_idx, output_idx, sizeof(size_t), cudaMemcpyDeviceToHost,
                        cuda_stream);
        cudaStreamSynchronize(cuda_stream);
    }

    alloc_cu->release(output_idx);
    return host_output_idx;
}
}  // namespace kernel
