#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

struct __align__(8) MD{
  float m;
  float d;
};

__device__ __forceinline__ MD reduce_md(MD a, MD b){
  MD result;
  bool a_bigger = a.m > b.m;
  MD bigger = a_bigger ? a : b;
  MD smaller = a_bigger ? b : a;
  result.m = bigger.m;
  result.d = bigger.d + smaller.d * __expf(smaller.m - bigger.m);
  return result;
}

template <int warpSize>
__device__ __forceinline__ MD warp_redcue_max(MD val){
#pragma unroll
  for(int mask = warpSize >> 1; mask >= 1; mask >>=1){
    MD other;
    other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
    other.d = __shfl_xor_sync(0xffffffff, val.d, mask);
    val = reduce_md(val, other);
  }
  return val;
}

template <int blockSize>
__global__ void online_softmax_kernel(const float* x, float*y, const int N){
  // m_i = max(m_i-1, x_i)
  // d_i = d_i-1 * exp(m_i-1 - m_i) + exp(x_i-m_i)
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane_id = tid % 32;
  constexpr int NUM_WARPS = blockSize / 32;

  const float* x_base = x + blockIdx.x * N;
  float* y_base = y + blockIdx.x * N;

  __shared__ MD shared_md[NUM_WARPS];

  MD patital;
  patital.m = -FLT_MAX;
  patital.d = 0.0f;
  // 每个thread 做一行
  for(int i = tid; i < N; i += blockSize){
    MD val;
    val.m = x_base[i];
    val.d = 1.0f;
    patital = reduce_md(patital, val);
  }

  // warp 内reduce
  MD reduce_res = warp_redcue_max<32>(patital);
  if(lane_id == 0){
    shared_md[warp_id] = reduce_res;
  }
  __syncthreads();
  // block 内
  MD res;
  res = lane_id < NUM_WARPS ? shared_md[lane_id] : MD{-FLT_MAX, 0.0f};
  res = warp_redcue_max<NUM_WARPS>(res);
  res.m = __shfl_sync(0xffffffff, res.m, 0);
  res.d = __shfl_sync(0xffffffff, res.d, 0);

  // 写回
  float max_val = res.m;
  float d_inv = 1.0f / res.d;
  for(int i = tid; i < N; i += blockSize){
    y_base[i] = d_inv * __expf(x_base[i] - max_val);
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