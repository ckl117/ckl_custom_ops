#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func))

// 声明在 .cu 文件中实现的函数
// void vector_add_cuda(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
// void gemm_simple(torch::Tensor& d, torch::Tensor& a, torch::Tensor& b);
// void gemm_multi_stage(torch::Tensor& d, torch::Tensor& a, torch::Tensor& b,
// c10::optional<torch::Tensor>& bias);
// void hgemm_mma_m16n8k16_mma2x4_warp4x4(torch::Tensor a, torch::Tensor b,
//                                        torch::Tensor c);
// void hgemm_mma_stages_block_swizzle_tn_cute(torch::Tensor a, torch::Tensor b,
//                                             torch::Tensor c, int stages,
//                                             bool swizzle, int swizzle_stride);
void online_softmax(torch::Tensor x, torch::Tensor y);

// void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y);
// void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x,
//                                               torch::Tensor y);
// 定义 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
  // m.def("gemm_simple", &gemm_simple, "GEMM One Stage (CUDA)");
  // m.def("gemm_multi_stage", &gemm_multi_stage, "GEMM Multi Stage (CUDA)");
  // TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_m16n8k16_mma2x4_warp4x4);
  // TORCH_BINDING_COMMON_EXTENSION(hgemm_mma_stages_block_swizzle_tn_cute);
  TORCH_BINDING_COMMON_EXTENSION(online_softmax);
  // TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token);
  // TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token);

}
