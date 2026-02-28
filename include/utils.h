#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)       \
  [&]() -> bool {                                                              \
    switch (pytorch_dtype) {                                                   \
    case at::ScalarType::Half: {                                               \
      using c_type = nv_half;                                                  \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case at::ScalarType::BFloat16: {                                           \
      using c_type = nv_bfloat16;                                              \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "           \
          << pytorch_dtype;                                                    \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, ...) \
  [&]() -> bool {                                                              \
    switch (pytorch_dtype) {                                                   \
    case at::ScalarType::Float: {                                              \
      using c_type = float;                                                    \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case at::ScalarType::Half: {                                               \
      using c_type = nv_half;                                                  \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case at::ScalarType::BFloat16: {                                           \
      using c_type = nv_bfloat16;                                              \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      std::ostringstream oss;                                                  \
      oss << __PRETTY_FUNCTION__ << " failed to dispatch data type "           \
          << pytorch_dtype;                                                    \
      TORCH_CHECK(false, oss.str());                                           \
      return false;                                                            \
    }                                                                          \
  }()
