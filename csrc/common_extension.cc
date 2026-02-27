#include <torch/extension.h>

// 声明在 .cu 文件中实现的函数
// void vector_add_cuda(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c);
void gemm_simple(torch::Tensor& d, torch::Tensor& a, torch::Tensor& b, c10::optional<torch::Tensor>& bias);
// void gemm_multi_stage(torch::Tensor& d, torch::Tensor& a, torch::Tensor& b, c10::optional<torch::Tensor>& bias);

// 定义 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
    m.def("gemm_simple", &gemm_simple, "GEMM One Stage (CUDA)");
    // m.def("gemm_multi_stage", &gemm_multi_stage, "GEMM Multi Stage (CUDA)");
}