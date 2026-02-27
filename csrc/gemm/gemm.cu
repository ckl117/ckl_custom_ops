#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>
#include "utils.h"

template <typename T, int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void gemm_simple_kernel(T *Cptr, const T *Aptr, const T *Bptr, const T *Biasptr, int m, int n, int k) {

  using namespace cute;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  int ix = blockIdx.x;
  int iy = blockIdx.y;

  Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
  Tensor gC = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));
  //  gA(kTileM, kTileK, num_tile_k)
  //  gB(kTileN, kTileK, num_tile_k)
  //  gC(kTileM, kTileN) 

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);
  // auto tAgA = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
  // auto tBgB = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
  // auto tCgC = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

  // auto tArA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
  // auto tBrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
  // auto tCrC = thr_mma.partition_fragment_C(gC(_, _));     // (MMA, MMA_M, MMA_N)
 
  // clear(tCrC);
  
//   int num_tile_k = size<2>(gA);
// #pragma unroll 1
//   for(int itile = 0; itile < num_tile_k; ++itile) {
//     cute::copy(tAgA(_, _, _, itile), tArA);
//     cute::copy(tBgB(_, _, _, itile), tBrB);

//     cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);
//   }

  // cute::copy(tCrC, tCgC); 
}

// 封装的 C++ 函数，供 Python 调用
void gemm_simple(torch::Tensor& d, torch::Tensor& a, torch::Tensor& b, c10::optional<torch::Tensor>& bias) {
    // 确保输入是 CUDA 张量且连续
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == b.dtype(), "a and b must have the same dtype");
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2, "mat_a must be a 2D tensor");
    TORCH_CHECK(b.dim() == 2, "mat_b must be a 2D tensor");
    TORCH_CHECK(a.stride(1) == 1, "mat_a must be a row major tensor");
    // TORCH_CHECK(b.stride(0) == 1, "mat_b must be a column major tensor");
    TORCH_CHECK(a.size(1) == b.size(1), "mat_a and mat_b shapes cannot be multiplied");
    auto m = a.size(0);
    auto k = a.size(1);
    auto n = b.size(0);
    if (bias) {
        TORCH_CHECK(bias->numel() == b.size(0), "size of bias is not matched");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dtype() == d.dtype(), "bias dtype must match output dtype");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    using namespace cute;
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, 
                        make_layout(Shape<_2, _2, _1>{}), 
                        make_layout(Shape<_1, _2, _1>{})));
    constexpr int kTileM = 128; 
    constexpr int kTileN = 128; 
    constexpr int kTileK = 32; 

    dim3 threads(size(MMA{}));
    // std::cout << "threads: " <<  threads.x << std::endl;
    dim3 blocks(n / kTileN, m / kTileM);
    // using scalar_t = nv_bfloat16;
    // using scalar_t = cute::bfloat16_t;
    // using scalar_t = cute::half_t;
    using scalar_t = __half;
    gemm_simple_kernel<scalar_t, kTileM, kTileN, kTileK, MMA><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<scalar_t*>(d.data_ptr()),
          reinterpret_cast<const scalar_t*>(a.data_ptr()),
          reinterpret_cast<const scalar_t*>(b.data_ptr()),
          bias ? reinterpret_cast<const scalar_t*>(bias->data_ptr()) : nullptr,
          m, n, k);
    // DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(a.scalar_type(), scalar_t, ([&] {
    //     gemm_simple_kernel<scalar_t, kTileM, kTileN, kTileK, MMA><<<blocks, threads, 0, stream>>>(
    //       static_cast<scalar_t*>(d.data_ptr()),
    //       static_cast<const scalar_t*>(a.data_ptr()),
    //       static_cast<const scalar_t*>(b.data_ptr()),
    //       bias ? static_cast<const scalar_t*>(bias->data_ptr()) : nullptr,
    //       m, n, k);
    //     return true;
    // }));

    // return out;
}


