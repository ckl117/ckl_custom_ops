// nvcc sm100_hgemm.cu -arch=compute_100a -code=sm_100a -lcuda -I/root/paddlejob/workspace/env_run/output/chenkailun/my/ckl_custom_ops/third_party/cutlass/include -o sm100_hgemm.o


#include "cuda.h"
#include "cuda_runtime.h"
#include <iostream>
#include <cuda_bf16.h>
#include <torch/library.h>

// #include <cute/atom/mma_traits_sm100.hpp>
// #include <cute/arch/mma_sm100_umma.hpp>
// #include <cute/arch/tmem_allocator_sm100.hpp>
// #include <cutlass/arch/barrier.h>
#include <cudaTypedefs.h>

#define LDST128Bits(x) (*reinterpret_cast<float4*>(&(x)) )

#define WARP_SIZE 32
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 256;

constexpr int TB_SIZE = 128;

// static CUtensorMapSwizzle mode_into_tensor_map_swizzle(const int& mode, const int& base) {
// #if CUDART_VERSION >= 12080
//     if (base != 0) {
//         // DG_HOST_ASSERT(base == 32 and mode == 128);
//         return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
//     }
// #endif

//     // DG_HOST_ASSERT(base == 0);
//     switch (mode) {
//         case   0:
//         case  16: return CU_TENSOR_MAP_SWIZZLE_NONE;
//         case  32: return CU_TENSOR_MAP_SWIZZLE_32B;
//         case  64: return CU_TENSOR_MAP_SWIZZLE_64B;
//         case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
//         default: printf("Unsupported swizzling mode\n");
//     }
// }

__device__  __host__ inline int 
ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

// https://github.com/NVIDIA/cutlass/blob/v4.3.1/include/cute/arch/mma_sm100_umma.hpp#L86
template <int CTA_GROUP = 1>
__device__ inline
void tcgen05_mma_f16(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, int enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"  // predicate register enable-input-d
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::%5.kind::f16 [%0], %1, %2, %3, p;\n\t"
    "}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d), "n"(CTA_GROUP)
  );
}

__device__ inline
void tma_2d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(mbar_addr) : "memory");
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__ inline
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__global__
void kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;
  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  extern __shared__ __align__(1024) char smem[]; // allocate shared memory
  const int A_smem = static_cast<int>(__cvta_generic_to_shared(smem));
  const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ uint64_t m_barriers[1];
  __shared__ int tmem_addr[1];  // tmem address is 32-bit

  const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(m_barriers));
  
  // only 1 thread initializes mbarrier
  if (tid == 0) {
    // initialize with 1, since only 1 thread issue TMA
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(1));
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }else if (warp_id == 1) {
    // one full warp allocates tmem
    // tcgen05.alloc returns tmem address in shared memory
    // -> we provide an smem address to the instruction.
    const int addr = static_cast<int>(__cvta_generic_to_shared(tmem_addr));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BLOCK_N));
  }
  __syncthreads();  // initialized mbarrier is visible to all threads in the threadblock

  // read tmem address, which has the value of 0.
  const int taddr = tmem_addr[0];

  int phase = 0;

  constexpr uint32_t i_desc = (1U << 4U)   // dtype=FP32
                            | (1U << 7U)   // atype=BF16
                            | (1U << 10U)  // btype=BF16
                            | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                            | ((uint32_t)BLOCK_M >> 4U << 24U)  // MMA_M
                            ;
  constexpr int MMA_K = 16;
  for(int iter_k = 0; iter_k < ceil_div(K, BLOCK_K); ++iter_k){
    if(tid == 0){
      for(int k = 0; k < BLOCK_K / 8; ++k){
        const int off_k = iter_k * BLOCK_K + k * 8;
        tma_2d_gmem2smem(A_smem + k * BLOCK_M * 16, &A_tmap, off_k, off_m, mbar_addr);
        tma_2d_gmem2smem(B_smem + k * BLOCK_N * 16, &B_tmap, off_k, off_n, mbar_addr);
      }
      constexpr int cp_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(cp_size) : "memory");
    }
    // wait for mbarrier to complete the current phase
    mbarrier_wait(mbar_addr, phase);

    // not sure if we need this. taken from DeepGEMM
    // https://github.com/deepseek-ai/DeepGEMM/blob/9b680f42/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh#L289
    asm volatile("tcgen05.fence::after_thread_sync;");

    // flip the phase parity, so that we wait for the correct parity in the next iteration
    phase ^= 1;

    if(tid == 0){
      auto make_desc = [](int addr, int height) -> uint64_t {
        const int LBO = height * 16;
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(LBO) << 16ULL) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };
      // manually unroll 1st iteration to disable accumulation
      tcgen05_mma_f16(taddr, make_desc(A_smem, BLOCK_M), make_desc(B_smem, BLOCK_N), i_desc, iter_k);
      for (int k = 1; k < BLOCK_K / MMA_K; k++) {
        // select MMA tile of A and B
        uint64_t a_desc = make_desc(A_smem + k * BLOCK_M * 32, BLOCK_M);
        uint64_t b_desc = make_desc(B_smem + k * BLOCK_N * 32, BLOCK_N);
        tcgen05_mma_f16(taddr, a_desc, b_desc, i_desc, 1);
      }
      // use the same mbarrier to track the completion of tcgen05.mma operations
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mbar_addr) : "memory");
    }

    // wait for MMA to finish
    mbarrier_wait(mbar_addr, phase);
    phase ^= 1;  // flip the phase

  }


  // this is required before tcgen05.ld and after tcgen05.mma
  // to form correct execution ordering.
  // see https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-canonical-sync-patterns-non-pipelined-diff-thread
  asm volatile("tcgen05.fence::after_thread_sync;");

  // load 8 columns from tmem at a time -> store 16 bytes per thread to smem
  for (int n = 0; n < BLOCK_N / 8; n++) {
    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d
    // Layout D
    float tmp[8];
    const int row = warp_id * 32;
    const int col = n * 8;
    const int addr = taddr + (row << 16) + col;  // 16 MSBs encode row, 16 LSBs encode column.
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]),
                  "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
                : "r"(addr));

    // wait for the completion of tcgen05.ld
    asm volatile("tcgen05.wait::ld.sync.aligned;");

    // cast and pack
    nv_bfloat162 out[4];
    for (int i = 0; i < 4; i++)
      out[i] = __float22bfloat162_rn({tmp[i * 2], tmp[i * 2 + 1]});

    // 16-byte per thread write (uncoalesced)
    nv_bfloat16 *out_ptr = C_ptr + (off_m + tid) * N + (off_n + n * 8);
    // if()
    reinterpret_cast<int4 *>(out_ptr)[0] = reinterpret_cast<int4 *>(out)[0];
  }

  // ensure all threads finish reading data from tmem before deallocation
  __syncthreads();
  if (warp_id == 0)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(BLOCK_N));

}

// forward declaration. we will cover this later.
void init_2D_tmap(
  CUtensorMap *tmap,
  const nv_bfloat16 *ptr,
  uint64_t gmem_outer_dim, uint64_t gmem_inner_dim,
  uint32_t smem_outer_dim, uint32_t smem_inner_dim
){
  int rank = 2;
  const cuuint64_t gmem_dims[2] = {static_cast<cuuint64_t>(gmem_inner_dim), static_cast<cuuint64_t>(gmem_outer_dim)};
  const cuuint64_t gmem_strides[1] = {static_cast<cuuint64_t>(gmem_inner_dim * sizeof(nv_bfloat16))};

  const cuuint32_t smem_dims[2] = {static_cast<cuuint32_t>(smem_inner_dim), static_cast<cuuint32_t>(smem_outer_dim)};
  const cuuint32_t elem_strides[2] = {1, 1};
  cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    rank,
    (void *)ptr,
    gmem_dims,
    gmem_strides,
    smem_dims,
    elem_strides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  
}

void matmul_v1a(
  const nv_bfloat16 *A_ptr,
  const nv_bfloat16 *B_ptr,
        nv_bfloat16 *C_ptr,
  int M, int N, int K
) {
  // prepare tensor map objects on the host.
  CUtensorMap A_tmap, B_tmap;
  init_2D_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, 8);
  init_2D_tmap(&B_tmap, B_ptr, N, K, BLOCK_N, 8);
  int grid = (M / BLOCK_M) * (N / BLOCK_N);

  // 1 A tile [BLOCK_M, BLOCK_K] and 1 B tile [BLOCK_N, BLOCK_K]
  const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);
  if (smem_size > 48000)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, C_ptr, M, N, K);
  cudaError_t ret = cudaGetLastError();
  if(ret != cudaSuccess){
    fprintf(stderr, "LAUNCH FAILED:%s\n", cudaGetErrorString(ret));
  }
}


int main(){
  using dtype = nv_bfloat16;
  int M = 128*4, N = 256 * 2, K = 256*2;
  int elem_size = sizeof(dtype);
  dtype* h_A = (dtype*)malloc(M * K * elem_size);
  dtype* h_B = (dtype*)malloc(N * K * elem_size);
  dtype* h_C = (dtype*)malloc(M * N * elem_size);
  dtype* h_C_ref = (dtype*)malloc(M * N * elem_size);
  for(int i = 0; i < M * K; i++){
    h_A[i] = __float2bfloat16(1);
  }
  for(int i = 0; i < N * K; i++){
    h_B[i] = __float2bfloat16(1);
  }
  h_A[0] = __float2bfloat16(0);
  h_B[0] = __float2bfloat16(0);

  dtype* d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, M * K * elem_size);
  cudaMalloc((void**)&d_B, N * K * elem_size);
  cudaMalloc((void**)&d_C, M * N * elem_size);
  cudaMemcpy(d_A, h_A, M * K * elem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * K * elem_size, cudaMemcpyHostToDevice);


  matmul_v1a(d_A, d_B, d_C, M, N, K);
  
  cudaMemcpy(h_C, d_C, M * N * elem_size, cudaMemcpyDeviceToHost);
  
  cudaError_t ret = cudaDeviceSynchronize();
  if(ret != cudaSuccess){
    fprintf(stderr, "last error:%s\n", cudaGetErrorString(ret));
  }
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < N; ++j){
      float acc = 0.f;
      for(int k = 0; k < K; ++k){
        acc += __bfloat162float(h_A[i * K + k] * h_B[j * K + k]);
      }
      h_C_ref[i * N + j] = __float2bfloat16(acc);
    }
  }

  float max_diff = 0.f;
  bool found_nan = false;
  

  for(int i = 0; i < M; ++i){
    for(int j = 0; j < N; ++j){
      float diff = abs(__bfloat162float(h_C[i * N + j]) - __bfloat162float(h_C_ref[i * N + j]));
      if (std::isnan(diff)) {
        found_nan = true;
      }
      if(diff > max_diff) max_diff = diff;
    }
  }
  std::cout << "found Nan: " << found_nan << " max diff: " << max_diff << std::endl;
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  
  return 0;
}