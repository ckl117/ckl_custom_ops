// nvcc tma.cu -arch=compute_90a -code=sm_90a -I /root/paddlejob/workspace/env_run/output/chenkailun/my/ckl_custom_ops/third_party/cutlass/include/ -o tma.o

#include <cute/tensor.hpp>

#define CHECK_CUDA2(func)                                              \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess) {                                   \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
        }                                                              \
    }

using namespace std;
using namespace cute;

template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
__global__ void tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor) {
    using namespace cute;
    constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);
    
    __shared__ T smem_data[CTA_M * CTA_N];
    __shared__ uint64_t tma_load_mbar;
    
    auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
    auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);
    
    if (threadIdx.x == 0) {
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
    
        auto gmem_tensor_coord_cta = local_tile(
            gmem_tensor_coord,
            Tile<Int<CTA_M>, Int<CTA_N>>{},
            make_coord(blockIdx.x, blockIdx.y));

        // if (cute::block(0)) {
        if (blockIdx.x == 0 && blockIdx.y == 0)
        cute::print_tensor(gmem_tensor_coord_cta);
        // }
    
        initialize_barrier(tma_load_mbar, /* arrival count */ 1);
    
        set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
    
        auto tma_load_per_cta = tma_load.get_slice(0);
        copy(tma_load.with(tma_load_mbar),
            tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
            tma_load_per_cta.partition_D(smem_tensor));
    }
    __syncthreads();
    wait_barrier(tma_load_mbar, /* phase */ 0);
    

    // 先return了！
    // return;


    // after this line, the TMA load is finished

    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("DEBUG SMEM\n");
        for (int i = 0; i < CTA_M; i++) {
            for (int j = 0; j < CTA_N; j++) {
                printf("%.0f ", smem_tensor(i, j));
            }
        printf("\n");
        }
        
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < CTA_M; i++) {
            for (int j = 0; j < CTA_N; j++) {
                smem_tensor(i, j) += smem_tensor(j, i);
            }
        }
        *(float*)(gmem_tensor.data().get()) = smem_tensor(0, 0);
    }
}

template <typename T, int CTA_M, int CTA_N>
void host_fn(T* data, int M, int N) {
    using namespace cute;
    
    T* ddata;
    CHECK_CUDA2(cudaMalloc(&ddata, sizeof(T) * M * N));
    CHECK_CUDA2(cudaMemcpy(ddata, data, sizeof(T) * M * N, cudaMemcpyHostToDevice));
    // create the GMEM tensor
    auto gmem_layout = make_layout(make_shape(M, N), LayoutRight{});
    auto gmem_tensor = make_tensor(make_gmem_ptr((T*)ddata), gmem_layout);
    using GmemTensor = decltype(gmem_tensor);
    
    // create the SMEM layout
    //! smem layout must be set during compile
    auto smem_layout = make_layout(make_shape(Int<CTA_M>{}, Int<CTA_N>{}), LayoutRight{});
    
    // create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);
    using tma_class = decltype(tma_load);
    
    // invoke the kernel
    tma_load_kernel<T, CTA_M, CTA_N, tma_class, GmemTensor>
                    <<<dim3{M / CTA_M, N / CTA_N, 1}, 32>>>
                    (tma_load, gmem_tensor);
    CHECK_CUDA2(cudaDeviceSynchronize());
    CHECK_CUDA2(cudaGetLastError());
}

int main() {
    #define data_M 4
    #define data_N 4
    float* data = (float*)malloc(data_M * data_N * sizeof(float));
    for (int i = 0; i < data_M * data_N; i++) {
        data[i] = i;
    }
    host_fn<float, 4, 4>(data, data_M, data_N);
}