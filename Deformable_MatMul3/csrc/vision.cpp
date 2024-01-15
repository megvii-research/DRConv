// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#ifndef vic
#define vic
#include "Deformable_MatMul.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &Deformable_MatMul_forward, "Deformable_MatMul_forward");
  m.def("backward", &Deformable_MatMul_backward, "Deformable_MatMul_backward");
}
#endif
