#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

template<const int kWarpSize=32>
__device__ __forceinline__ float warp_reduce_max(float val){
#pragma unroll
  for(int mask = kWarpSize>>1; mask >= 1; mask >>=1){
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<const int kWarpSize=32>
__device__ __forceinline__ float warp_reduce_sum(float val){
#pragma unroll
  for(int mask = kWarpSize >> 1; mask >= 1; mask >>=1){
    val = val +  __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

template<const int NUM_THREADS=128>
__global__ void sate_softmax_kernel(const float* input, float* output, int cols) {
  // 每个block处理一行, 每个线程处理一个元素
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  const float* input_base = input + blockIdx.x * cols;
  float* output_base = output + blockIdx.x * cols;

  float max_val = -FLT_MAX;
  // block内的最大线程数1024
  __shared__ float shared_data[32];
  if(tid < 32){
    shared_data[tid] = -FLT_MAX;
  }

  for(int i = tid; i < cols; i += blockDim.x){
    max_val = fmaxf(max_val, input_base[i]);
  }
  // warp reduce max_val
  max_val = warp_reduce_max(max_val);
  // block reduce max_val
  if (lane_id == 0){
    shared_data[warp_id] = max_val;
  }
  __syncthreads();
  if(tid < 32){
    max_val = shared_data[tid];
    max_val = warp_reduce_max(max_val);
    if(tid == 0){
      shared_data[0] = max_val;
    }
  }
  __syncthreads();
  max_val = shared_data[0];

  float exp_val = 0;
  float sum_val = 0;
  for(int i = tid; i < cols; i += blockDim.x){
    exp_val = __expf(input[i] - max_val);
    sum_val += exp_val;
  }
  // warp reduce sum_val
  sum_val = warp_reduce_sum(sum_val);
  if (lane_id == 0){
    shared_data[warp_id] = sum_val;
  }
  __syncthreads();
  if (tid < NUM_WARPS){
    shared_data[tid] = sum_val;

  }
  sum_val = lane_id < NUM_WARPS ? shared_data[lane_id] : 0.f;
  sum_val = warp_reduce_sum<NUM_WARPS>(sum_val);
  sum_val = __shfl_sync(0xffffffff, sum_val, 0);

  for(int i = tid; i < cols; i += blockDim.x){
    output_base[i] = __expf(input_base[i] - max_val) / sum_val;
  }

}



struct __align__(8) MD{
  float m;
  float d;
};

__device__ __forceinline__ MD reduce_md(MD a, MD b){
  bool a_bigger = a.m > b.m;
  MD bigger = a_bigger ? a : b;
  MD smaller = a_bigger ? b : a;
  MD res;
  res.m = bigger.m;
  res.d = bigger.d + smaller.d * __expf(smaller.m - bigger.m);
  return res;
}
template<const int kWarpSize=32>
__device__ __forceinline__ MD warp_reduce_md(MD val){
#pragma unroll
  for(int mask = kWarpSize >> 1; mask >= 1; mask >>= 1){
    MD other;
    other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
    other.d = __shfl_xor_sync(0xffffffff, val.d, mask);
    val = reduce_md(val, other);
  }
  return val;
}

template<const int NUM_THREADS=128>
__global__ void online_softmax_kernel(const float* input, float* output, int cols) {
  // m_0 = x0, d_0 = 1.0
  // m_i = max(m_i-1, x_i)
  // d_i = exp(x_i - m_i) + d_i-1 * exp(m_i-1 - m_i)
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  constexpr int NUM_WARPS = NUM_THREADS / 32;

  const float* input_base = input + blockIdx.x * cols;
  float* output_base = output + blockIdx.x * cols;
  __shared__ MD shared_data[NUM_WARPS];
  
  MD partial;
  partial.m = -FLT_MAX;
  partial.d = 0.0f;
  for(int i = tid; i < cols; i += blockDim.x){
    MD other;
    other.m = input_base[i];
    other.d = 1.0f;
    partial = reduce_md(partial, other);
  }

  // warp reduce
  MD res;
  res = warp_reduce_md(partial);
  if(lane_id == 0){
    shared_data[warp_id] = res;
  }
  __syncthreads();
  // block reduce
  partial = lane_id < NUM_WARPS ? shared_data[lane_id] : MD{-FLT_MAX, 0.f};
  partial = warp_reduce_md<NUM_WARPS>(partial);
  partial.m = __shfl_sync(0xffffffff, partial.m, 0);
  partial.d = __shfl_sync(0xffffffff, partial.d, 0);

  float inv_exp_sum = 1.f / partial.d;
  for(int i = tid; i < cols; i += blockDim.x){
    output_base[i] = inv_exp_sum * __expf(input_base[i] - partial.m);
  }
}

void online_softmax(torch::Tensor x, torch::Tensor y){
  TORCH_CHECK(x.dtype() == torch::kFloat32);
  TORCH_CHECK(x.dim() == 2);
  TORCH_CHECK(y.dim() == 2);
  TORCH_CHECK(x.size(0) == y.size(0));
  TORCH_CHECK(x.size(1) == y.size(1));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(128);
  dim3 grid(x.size(0));

  online_softmax_kernel<128><<<grid, block, 0, stream>>>(
    reinterpret_cast<const float*>(x.data_ptr()),
    reinterpret_cast<float*>(y.data_ptr()),
    x.size(1));

}