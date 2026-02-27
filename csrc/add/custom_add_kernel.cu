#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA 核函数声明
template <typename T>
__global__ void vector_add_kernel(const T* a, const T* b, T* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

// 封装的 C++ 函数，供 Python 调用
void vector_add_cuda(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c) {
    // 确保输入是 CUDA 张量且连续
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == b.dtype(), "a and b must have the same dtype");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int n = a.numel();

    // 根据数据类型分派核函数
    dim3 threads(256);
    dim3 blocks((n + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "vector_add_cuda", ([&] {
        vector_add_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            n
        );
    }));

    // return out;
}