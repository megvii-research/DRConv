import torch
import pickle
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from Deformable_MatMul3.Deformable_MatMul3 import Deformable_MatMul3

class DRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, dilation=1, groups_num=8, num_W=8):
        super(DRConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.groups_num = groups_num
        self.num_W = num_W
        self.use_unflod = False if (self.kernel_size[0] == 1 and self.stride[0] == 1 and self.padding[0] == 0 and self.dilation[0] == 1) else True

        self.weight = nn.Parameter(torch.Tensor(out_channels * in_channels * self.kernel_size[0] * self.kernel_size[1], self.num_W))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.routing_fc = nn.Linear(in_channels, self.groups_num * self.num_W)
        self.deformable_matmul3 = Deformable_MatMul3()

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        bound = math.sqrt(6/n)
        self.weight.data.uniform_(-bound, bound)
        if self.bias is not None:
            self.bias.data.zero_()

        self.routing_fc.weight.data.normal_(0, 0.01)
        if self.routing_fc.bias is not None:
            self.routing_fc.bias.data.zero_()


    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs, mask, Alpha, use_alpha=False, beta=1.0):
        alpha = 1.0

        batch_size, channel, height, width = inputs.shape

        self.out_height = int(math.floor(
            (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
        self.out_width = int(math.floor(
            (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))

        x_se = inputs.reshape(inputs.shape[0], inputs.shape[1], -1).mean(dim=-1, keepdim=False)
        x_se = 2.0 * F.sigmoid(alpha * self.routing_fc(x_se)) / self.num_W
        weight = F.linear(x_se.reshape(-1, self.num_W), self.weight)
        weight = weight.reshape(batch_size, self.groups_num, self.out_channels, self.in_channels * self.kernel_size[0] * self.kernel_size[1]).permute(0, 2, 3, 1)

        if self.use_unflod:
            # N x [inC * kH * kW] x [outH * outW]
            inputs = F.unfold(inputs, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride) # n, c*k*k, oh*ow
            inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], self.out_height, self.out_width) # n, c*k*k, oh*ow

        if mask.shape[-2:] != torch.Size([self.out_height, self.out_width]):
            mask = F.interpolate(mask.float().unsqueeze(1), (self.out_height, self.out_width)).squeeze(1)
            Alpha = F.interpolate(Alpha, (self.out_height, self.out_width))
        out = self.deformable_matmul3(inputs, weight, mask, Alpha, use_alpha, beta) # weight:  [n,] o, ckk, groups_num    out: n, o, oh, ow

        if self.bias is not None:
            out = out + self.bias.reshape(1, -1, 1, 1)
        return out

