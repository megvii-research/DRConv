import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import Deformable_MatMul_cu3

class _Deformable_MatMul3(Function):
    @staticmethod
    def forward(ctx, mat0, mat1, mask, Alpha, use_alpha, beta):
        mask = mask.to(mat0.dtype)
        ctx.save_for_backward(mat0, mat1, mask, Alpha)
        ctx.batch, ctx.input_channel, ctx.height, ctx.width = mat0.shape
        ctx.batch_W = ctx.batch
        ctx.beta = beta
        ctx.use_alpha = 1 if use_alpha else 0

        if len(mask.shape) == 3:
            ctx.mask_num = mask.shape[0]
        else:
            ctx.mask_num = 1
        if len(mat1.shape) == 3:
            mat1 = mat1.unsqueeze(0)
            ctx.batch_W = 1
        _, ctx.output_channel, _, ctx.num = mat1.shape
        output = Deformable_MatMul_cu3.forward(mat0.contiguous(), mat1.contiguous(), mask.contiguous(), ctx.batch, ctx.input_channel, ctx.height, ctx.width, ctx.output_channel, ctx.num, ctx.mask_num, ctx.batch_W)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        mat0, mat1, mask, Alpha = ctx.saved_tensors
        if len(mat1.shape) == 3:
            mat1 = mat1.unsqueeze(0)
        grad_mat0, grad_mat1, grad_Alpha = Deformable_MatMul_cu3.backward(grad_output.contiguous(), mat0.contiguous(), mat1.contiguous(), mask.contiguous(), Alpha.contiguous(), ctx.batch, ctx.input_channel, ctx.height, ctx.width, ctx.output_channel, ctx.num, ctx.mask_num, ctx.batch_W, ctx.use_alpha)
        return grad_mat0, grad_mat1, None, ctx.beta * grad_Alpha, None, None


deformable_matmul3 = _Deformable_MatMul3.apply

class Deformable_MatMul3(nn.Module):
    def __init__(self):
        super(Deformable_MatMul3, self).__init__()

    def forward(self, mat0, mat1, mask, Alpha, use_alpha=False, beta=1.0):
        if torch.cuda.is_available():
            return deformable_matmul3(mat0, mat1, mask, Alpha, use_alpha, beta)
        else:
            batch, _, h, w = mat0.shape
            out_channel = mat1.shape[0] if len(mat1.shape) == 3 else mat1.shape[1]
            return torch.zeros(batch, out_channel, h, w)


