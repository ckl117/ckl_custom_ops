
// #include <ATen/Operators.h>
// #include <torch/all.h>
// #include <torch/library.h>
// #include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#define WARP_SIZE 32

__global__ void kernel_shuffle(){
    const int idx = threadIdx.x;
    float val = idx;
    __syncthreads();
    printf("Init thread %d: %f\n", idx, val);

    // val = __shfl_sync(0xffffffff, val, 0, WARP_SIZE / 4);
    // __syncthreads();
    // printf("__shfl_sync thread %d: %f\n", idx, val);
    // val = idx;
    // __syncthreads();
    // val = __shfl_sync(0xffff0000, val, 1, WARP_SIZE);
    // __syncthreads();
    // printf("__shfl_sync thread %d: %f\n", idx, val);

    for(int offset = 16; offset > 1; offset >> 1){

        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    __syncthreads();
    printf("__shfl_up_sync thread %d: %f\n", idx, val);
    
    // val = idx;
    // val = __shfl_up_sync(0xffffffff, val, 1, WARP_SIZE / 2);
    // __syncthreads();
    // printf("__shfl_up_sync thread %d: %f\n", idx, val);
}

int main(){
    kernel_shuffle<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}

// void shuffle_cuda() {
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   kernel_shuffle<<<1, 32, 0, stream>>>();
// }
