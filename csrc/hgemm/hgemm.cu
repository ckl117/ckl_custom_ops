#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include <cuda_fp16.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define LOAD_128bit(x) ((reinterpret_cast<float4*>(&(x)))[0])

template<typename T,
const int TILE_M=16, const int TILE_N=8, const int TILE_K=16,
const int MMA_M=16, const int MMA_N=8, const int MMA_K=16>
__device__ void hgemm_kernel(T* __restrict__ A, T* __restrict__ B, T* __restrict__ C,
                            int M, int N, int K) {
    const int tid = threadIdx.x;
    const int bid_x = blockIdx.x;
    const int bid_y = blockIdx.y;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    extern __shared__ T shared_data[];
    T* sA = shared_data;
    T* sB = shared_data + TILE_M * TILE_K;
    constexpr int THREAD_TILE_K = TILE_K * sizeof(T) / 16;
    static_assert
    const int num_rows_in_block_A = blockDim.x / THREAD_TILE_K;
    assert(blockDim.x % THREAD_TILE_K == 0);
    const int offset_m_global = blockIdx.x * TILE_M;
    const int offset_n_global = blockIdx.y * TILE_N;
    half reg_a[8];
    half reg_b[4];
    half reg_c[4];

    for(int k=0; k<K; k+=TILE_K) {

        // load A and B into shared memory
        for(int start_row_id = 0; start_row_id < TILE_M; start_row_id+=num_rows_in_block_A) {
            const int offset_k_in_tile = tid % THREAD_TILE_K;
            const int offset_m_in_tile = start_row_id + tid / THREAD_TILE_K;
            LOAD_128bit(sA[offset_m_in_tile * TILE_K + offset_k_in_tile]) = LOAD_128bit(A[(offset_m_global + offset_m_in_tile) * N + k + offset_k_in_tile]);
            half b = B[K * TILE_K + lane_id][tid];
        }
        for(int start_n_id = 0; start_n_id < TILE_N; start_n_id+=num_rows_in_block_A) {
            const int offset_k_in_tile = tid % THREAD_TILE_K;
            const int offset_n_in_tile = start_n_id + tid / THREAD_TILE_K;
            if (offset_n_in_tile >= TILE_N) continue
            LOAD_128bit(sB[offset_n_in_tile * TILE_K + offset_k_in_tile]) = LOAD_128bit(B[(offset_n_global + offset_n_in_tile) * K + k + offset_k_in_tile]);
            half b = B[K * TILE_K + lane_id][tid];
        }
        __syncthreads();

        // smem -> reg
        

    }
}

void launch_hgemm() {
    // mma = 16, 8, 16, 单个MMA占用smem (16*16+16*8)*2B = 768 B
    // 8 16, 
    // (16*16+16*8)*2B
    
    int M = 1024;
    int N = 1024;
    int K = 1024;

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 8;
    constexpr int TILE_K = 16;

    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;

    // 计算需要的threadblock数量
    dim3 block_size(32);
    dim3 grid_dim(CEIL_DIV(M, TILE_M), CEIL_DIV(N, TILE_N));
    int shared_mem_size = (TILE_M * TILE_N + TILE_M * TILE_K) * sizeof(half);
    
    hgemm_kernel<TILE_M, TILE_N, TILE_K, MMA_M, MMA_N, MMA_K><<<grid_dim, block_size, shared_mem_size>>>(nullptr, nullptr, nullptr,M, N, K);

    return ;
}

int main() {
    int deviceId = 0; // 假设我们使用第一个设备
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    std::cout << "设备名称: " << prop.name << std::endl;
    std::cout << "每个线程块（Block）的共享内存大小 (sharedMemPerBlock): "
              << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块可选的共享内存大小 (sharedMemPerBlockOptin): "
              << prop.sharedMemPerBlockOptin / 1024.0 << " KB" << std::endl;
    std::cout << "每个SM的共享内存大小 (sharedMemPerMultiprocessor): "
              << prop.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;

    return 0;
}