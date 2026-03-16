#include "../cpu/mha_kernel.h"
#include "../kernels_interface.h"
namespace kernel {
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                base::DeviceType device_type, CudaConfig* config) {
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float scale = 1.f / std::sqrt(static_cast<float>(head_size));
    std::shared_ptr<base::DeviceAllocator> allocator;
    if (device_type == base::DeviceType::kDeviceCPU) {
        allocator = base::CPUDeviceAllocatorFactory::get_instance();
    } else {
        allocator = base::CUDADeviceAllocatorFactory::get_instance();
    }

    auto make_vector_view = [device_type](int32_t size, float* data) {
        auto buffer = std::make_shared<base::Buffer>(size * sizeof(float), nullptr, data, true,
                                                     device_type);
        tensor::Tensor tensor(base::DataType::kDataTypeFp32, size);
        CHECK(tensor.assign(buffer));
        return tensor;
    };

    auto make_matrix_view = [device_type](int32_t dim0, int32_t dim1, float* data) {
        auto buffer = std::make_shared<base::Buffer>(dim0 * dim1 * sizeof(float), nullptr, data,
                                                     true, device_type);
        tensor::Tensor tensor(base::DataType::kDataTypeFp32, dim0, dim1);
        CHECK(tensor.assign(buffer));
        return tensor;
    };

    for (int32_t h = 0; h < head_num; ++h) {
        float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
        float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);
        tensor::Tensor query_mat = make_vector_view(head_size, query_head_addr);

        for (int32_t t = 0; t <= pos; t++) {
            int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
            const float* key_head_addr =
                key_cache_tensor.ptr<float>() + layer_offset + cache_offset;
            tensor::Tensor key_mat = make_matrix_view(1, head_size, const_cast<float*>(key_head_addr));
            tensor::Tensor score_mat = make_vector_view(1, score_head_addr + t);
            get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
        }

        tensor::Tensor score_head_tensor = make_vector_view(pos + 1, score_head_addr);
        get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);
        float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
        allocator->memset_zero(output_head_ptr, sizeof(float) * head_size,
                               config ? config->stream : nullptr, false);
        tensor::Tensor output_tensor = make_vector_view(head_size, output_head_ptr);
        int32_t cache_offset = (h / kv_mul) * head_size;
        float* value_head_addr =
            const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
        tensor::Tensor value_tensor = make_vector_view(head_size, value_head_addr);
        get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos,
                                          head_size, kv_dim, config ? config->stream : nullptr);
    }
}
}  // namespace kernel
