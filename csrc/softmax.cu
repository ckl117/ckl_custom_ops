#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cub/cub.cuh>

struct __align__(8) mMD {
  float m;
  float d;
};

__device__ __forceinline__ mMD reduce_md(mMD a, mMD b){
  bool a_bigger = a.m > b.m;
  mMD bigger = a_bigger ? a : b;
  mMD smaller = a_bigger ? b : a;
  mMD res;
  res.m = bigger.m;
  res.d = bigger.d + smaller.d * __expf(smaller.m - bigger.m);
  return res;
}
template<const int kWarpSize=32>
__device__ __forceinline__ mMD warp_reduce_md(mMD val){
# pragma unroll
  for(int mask = kWarpSize >> 1; mask >= 1; mask >>= 1){
    mMD other;
    other.m = __shfl_xor_sync(0xffffffff, val.m, mask);
    other.d = __shfl_xor_sync(0xffffffff, val.d, mask);
    val = reduce_md(val, other);
  }
  return val;
}

template<const int NUM_THREADS=128>
__global__ void online_softmax_kernel(const float* x, float* y, const int cols){
  const int tid = threadIdx.x;
  // init m_0 = x_i, d_0 = 1.0
  // m_i = max(m_(i-1), x_i)
  // d_i = d_(i-1)*exp(m_(i-1) - m_i) + exp(x_i - m_i)

  __shared__ mMD md_shared[32];
  const int warp_id = tid / 32;
  const int warp_lane = tid % 32;
  const int NUM_WARPS = NUM_THREADS / 32;
  if (warp_id == 0){
    md_shared[warp_lane].m = -FLT_MAX;
    md_shared[warp_lane].d = 0.0;
  }
  __syncthreads();

  const float* x_base_ptr = x + blockIdx.x * cols;
  float* y_base_ptr = y + blockIdx.x * cols;

  mMD md_partial;
  md_partial.m = -FLT_MAX;
  md_partial.d = 0.0;
  for(int i = tid; i < cols; i += NUM_THREADS){
    mMD md_new;
    md_new.m = x_base_ptr[i];
    md_new.d = 1.0;
    md_partial = reduce_md(md_partial, md_new);
  }
  mMD res;
  res = warp_reduce_md(md_partial);
  if (warp_lane == 0){
    md_shared[warp_id] = res;
  }
  __syncthreads();
  // Do block reduce
  if (tid < 32){
    mMD block_res = md_shared[tid];
    block_res = warp_reduce_md<NUM_WARPS>(block_res);
    if (tid == 0){
      md_shared[0] = block_res;
    }
  }
  __syncthreads();

  float d_inv = __fdividef(1.0, md_shared[0].d);
  float max_val = md_shared[0].m;
  for(int i = tid; i < cols; i += NUM_THREADS){
    y_base_ptr[i] = __expf(x_base_ptr[i] - max_val) * d_inv;
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