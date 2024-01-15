// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#ifndef CJCJCJ
#define CJCJCJ
#pragma once
//#include <torch/extension.h>
#include <vector>
#include <torch/types.h>

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


torch::Tensor Deformable_MatMul_forward(torch::Tensor mat0, torch::Tensor mat1, torch::Tensor mask,
                                const int batch,
                                const int input_channel,
                                const int height,
                                const int width,
                                const int output_channel,
                                const int num, 
                                const int mask_num,
                                const int batch_W) {
  if (mat0.type().is_cuda()) {
#ifdef WITH_CUDA
    return Deformable_MatMul_forward_cuda(mat0, mat1, mask, batch, input_channel, height, width, output_channel, num, mask_num, batch_W);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

std::vector<torch::Tensor> Deformable_MatMul_backward(torch::Tensor grad_output, torch::Tensor mat0, torch::Tensor mat1, torch::Tensor mask, torch::Tensor Alpha,
                                const int batch,
                                const int input_channel,
                                const int height,
                                const int width,
                                const int output_channel,
                                const int num, 
                                const int mask_num,
                                const int batch_W,
                                const int use_alpha) {
  if (grad_output.type().is_cuda()) {
#ifdef WITH_CUDA
    return Deformable_MatMul_backward_cuda(grad_output, mat0, mat1, mask, Alpha, batch, input_channel, height, width, output_channel, num, mask_num, batch_W, use_alpha);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
#endif



