#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


__global__ void test_kernel(float* a){
    const int tid = threadIdx.x;
    float q_left = a[0];
    float q_right = a[1];
    float rotary_embs_cos = a[2];
    float rotary_embs_sin = a[3];
    float rope_left = 0.f;
    float rope_right = 0.f;
    // -fmad=true 结果 0.10815426
    // -fmad=false 0.1081543
    if (tid == 0){
        printf("q_left: %.8g, q_right: %.8g, rotary_embs_cos: %.8g, rotary_embs_sin: %.8g\n", q_left, q_right, rotary_embs_cos, rotary_embs_sin);
        rope_left = __fmul_rn(q_left, rotary_embs_cos) - __fmul_rn(q_right, rotary_embs_sin);
        rope_right = __fmul_rn(q_right, rotary_embs_cos) + __fmul_rn(q_left, rotary_embs_sin);
        printf("rope_left: %.8g, rope_right: %.8g\n", rope_left, rope_right);
        // nvcc 默认-fmad=true, a*b+c 仅对结果做rn
        rope_left = q_left* rotary_embs_cos - q_right * rotary_embs_sin;
        rope_right = q_right * rotary_embs_cos + q_left * rotary_embs_sin;
        printf("rope_left: %.8g, rope_right: %.8g\n", rope_left, rope_right);
    }
}

int main(){
    float* h_a = (float*)malloc(10 * sizeof(float));
    h_a[0] = 1.71093750f;
    h_a[1] = -1.56250000f;
    h_a[2] = 0.70613062f;
    h_a[3] = 0.7080816f;

    float *d_a ;
    cudaMalloc(&d_a, 10 * sizeof(float));
    cudaMemcpy(d_a, h_a, 10 * sizeof(float), cudaMemcpyHostToDevice);

    test_kernel<<<1,32>>>(d_a);
    cudaDeviceSynchronize();
    return 0;
}