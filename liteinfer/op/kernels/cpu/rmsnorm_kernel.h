#ifndef LLAMA_INFER_RMSNORM_KERNEL_H
#define LLAMA_INFER_RMSNORM_KERNEL_H
#include "base/tensor.h"

namespace kernel {
void rmsnorm_kernel_cpu(const tensor::Tensor& input, const tensor::Tensor& weight,
                        const tensor::Tensor& output, float eps, void* stream = nullptr);
}  // namespace kernel
#endif  // LLAMA_INFER_RMSNORM_KERNEL_H
