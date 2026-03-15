#include "rope_kernel.cuh"
namespace kernel {
namespace {
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) {
    float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
    float2 vec_value = *vec_ptr;
    *vec_ptr =
        make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(bool use_half_split, int pos, int dim, int kv_dim,
                                    int head_size, const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
    const int linear_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (use_half_split) {
        const int num_heads = dim / head_size;
        const int head_pair_count = head_size / 2;
        const int total_pairs = num_heads * head_pair_count;
        if (linear_idx >= total_pairs) {
            return;
        }

        const int head_idx = linear_idx / head_pair_count;
        const int head_dim = linear_idx % head_pair_count;
        const int i = head_idx * head_size;
        const int v0_idx = i + head_dim;
        const int v1_idx = i + head_dim + head_size / 2;

        const float fci = sin_cache[pos * head_size + head_dim * 2];
        const float fcr = cos_cache[pos * head_size + head_dim * 2];
        const int rotn = i < kv_dim ? 2 : 1;

        for (int v = 0; v < rotn; v++) {
            float* vec = const_cast<float*>(v == 0 ? input_q : input_k);
            const float v0 = vec[v0_idx];
            const float v1 = vec[v1_idx];
            vec[v0_idx] = fcr * v0 - fci * v1;
            vec[v1_idx] = fcr * v1 + fci * v0;
        }
        return;
    }

    const int idx = linear_idx * 2;
    if (idx >= dim) {
        return;
    }

    const int head_dim = idx % head_size;
    const float fci = *(sin_cache + pos * head_size + head_dim);
    const float fcr = *(cos_cache + pos * head_size + head_dim);

    rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
    if (idx >= kv_dim) {
        return;
    }
    rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

__global__ void sin_cos_calc(float rope_theta, int head_size, int max_seq_len, float* sin_cache,
                             float* cos_cache) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int head_dim = idx % head_size;
    for (int pos = 0; pos < max_seq_len; ++pos) {
        float freq =
            1.0f / powf(rope_theta, static_cast<float>(head_dim) / static_cast<float>(head_size));
        float val = static_cast<float>(pos) * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        *(sin_cache + pos * head_size + head_dim) = fci;
        *(cos_cache + pos * head_size + head_dim) = fcr;
    }
}
}  // namespace

void sin_cos_cache_calc_cu(base::ModelType model_type, int head_size, int max_seq_len,
                           const tensor::Tensor& sin_cache, const tensor::Tensor& cos_cache,
                           cudaStream_t stream) {
    CHECK_EQ(sin_cache.is_empty(), false);
    CHECK_EQ(cos_cache.is_empty(), false);
    const int threads = head_size;
    const float rope_theta = base::RoPETheta(model_type);
    if (stream) {
        sin_cos_calc<<<1, threads, 0, stream>>>(rope_theta, head_size, max_seq_len,
                                                const_cast<float*>(sin_cache.ptr<float>()),
                                                const_cast<float*>(cos_cache.ptr<float>()));
    } else {
        sin_cos_calc<<<1, threads>>>(rope_theta, head_size, max_seq_len,
                                     const_cast<float*>(sin_cache.ptr<float>()),
                                     const_cast<float*>(cos_cache.ptr<float>()));
    }
}

void rope_kernel_cu(base::ModelType model_type, int32_t dim, int32_t kv_dim, int32_t head_size,
                    const tensor::Tensor& input_q, const tensor::Tensor& input_k,
                    const tensor::Tensor& input_pos, const tensor::Tensor& sin_cache,
                    const tensor::Tensor& cos_cache, void* stream) {
    const int32_t pos = *input_pos.ptr<int32_t>(0);
    const bool use_half_split = base::UsesHalfSplitRoPE(model_type);
    const int total_work = use_half_split ? (dim / head_size) * (head_size / 2) : (dim + 1) / 2;
    if (total_work <= 0) {
        return;
    }
    const int threads = 128;
    const int blocks = (total_work + threads - 1) / threads;
    if (stream) {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
            use_half_split, pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
            sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } else {
        rope_kernel_cu_fp32<<<blocks, threads>>>(use_half_split, pos, dim, kv_dim, head_size,
                                                 input_q.ptr<float>(), input_k.ptr<float>(),
                                                 sin_cache.ptr<float>(), cos_cache.ptr<float>());
    }
}
}  // namespace kernel
