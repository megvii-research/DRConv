// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#ifndef tess
#define tess
#include <torch/types.h>
#pragma once
//#include <torch/extension.h>
#include <vector>

torch::Tensor Deformable_MatMul_forward_cuda(torch::Tensor mat0, torch::Tensor mat1, torch::Tensor mask,
                                const int batch,
                                const int input_channel,
                                const int height,
                                const int width,
                                const int output_channel,
                                const int num, 
                                const int mask_num,
                                const int batch_W);

std::vector<torch::Tensor> Deformable_MatMul_backward_cuda(torch::Tensor grad_output, torch::Tensor mat0, torch::Tensor mat1, torch::Tensor mask, torch::Tensor Alpha,
                                const int batch,
                                const int input_channel,
                                const int height,
                                const int width,
                                const int output_channel,
                                const int num, 
                                const int mask_num,
                                const int batch_W,
                                const int use_alpha);
#endif
