import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
from DRConv import DRConv

from opCounter import profile
from opCounter.utils import clever_format
from opCounter.count_hooks import count_Deformable_for_Dynamic_localshare_via_automask_gradient_Conv2d

class LConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                                 padding=0, dilation=1, groups=1, bias=True):
        super(LConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                             padding, dilation, groups, bias)
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class conv_bn_relu(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding,
            groups=1, bias=True, bn=True, relu=True, DLS=False, groups_num=8):
        super(conv_bn_relu, self).__init__()
        self.DLS = DLS
        self.has_bn = bn
        self.has_relu = relu

        if DLS:
            self.conv = DRConv(input_channel, output_channel, kernel_size, stride, padding, groups=groups, bias=bias, groups_num=groups_num, num_W=8)
        else:
            self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, groups=groups, bias=bias)

        if self.has_bn:
            self.bn = nn.BatchNorm2d(output_channel, momentum=0.1)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask=None, Alpha=None):
        if self.DLS:
            x = self.conv(x, mask, Alpha)
        else:
            x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

def channel_shuffle_v2(x):
    batchsize, num_channels, height, width = x.data.size()

    # reshape
    x = x.reshape(batchsize * num_channels // 2,
        2, height * width)
    x = x.permute(1, 0, 2)
    # flatten
    x = x.reshape(2, batchsize, num_channels // 2, height, width)

    return x[0], x[1]

class make_block(nn.Module):
    def __init__(self, inp, oup, stride, groups_num=8):
        super(make_block, self).__init__()
        self.stride = stride
        self.inp = inp
        self.groups_num = groups_num
        assert stride in [1, 2]

        inp = inp if self.stride == 2 else inp // 2
        right_oup = oup - inp
        mid_channel = oup // 2

        self.masknet = nn.Conv2d(inp, groups_num, kernel_size=1, stride=1, padding=0, bias=False)
        self.conva_right = conv_bn_relu(inp, mid_channel, kernel_size=1, stride=1, padding=0, bias=False, DLS=True, groups_num=groups_num)
        self.convb_right = conv_bn_relu(mid_channel, mid_channel, kernel_size=3, stride=self.stride, padding=1, groups=mid_channel, bias=False, relu=False)
        self.convc_right = conv_bn_relu(mid_channel, right_oup, kernel_size=1, stride=1, padding=0, bias=False, DLS=True, groups_num=groups_num)

        if self.stride == 2:
            self.conva_left = conv_bn_relu(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp, bias=False, relu=False)
            self.convb_left = conv_bn_relu(inp, inp, kernel_size=1, stride=1, padding=0, bias=False, DLS=False, groups_num=groups_num)

    def forward(self, x, alpha=1.0):
        if self.stride == 2:
            x_proj = x
            x_proj = self.conva_left(x_proj)
            x_proj = self.convb_left(x_proj)
        else:
            x_proj, x = channel_shuffle_v2(x)

        Alpha = self.masknet(x)
        Alpha =  F.softmax(Alpha, dim=1)
        mask = torch.argmax(Alpha, dim=1)
        x = self.conva_right(x, mask, Alpha)
        x = self.convb_right(x)
        x = self.convc_right(x, mask, Alpha)

        return torch.cat((x_proj, x), 1)


class ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(ShuffleNetV2, self).__init__()

        assert input_size % 32 == 0

        if width_mult == 0.5:
            channels = [24, 48, 96, 192, 1024]
        elif width_mult == 1.0:
            channels = [24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            channels = [24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            channels = [24, 244, 488, 976, 2048]

        print('train shufflev2 {}x'.format(width_mult))

        self.conv_first = conv_bn_relu(input_channel=3, output_channel=channels[0], kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1_block1 = make_block(channels[0], channels[1], 2)
        self.stage1_block2 = make_block(channels[1], channels[1], 1)
        self.stage1_block3 = make_block(channels[1], channels[1], 1)
        self.stage1_block4 = make_block(channels[1], channels[1], 1)

        self.stage2_block1 = make_block(channels[1], channels[2], 2)
        self.stage2_block2 = make_block(channels[2], channels[2], 1)
        self.stage2_block3 = make_block(channels[2], channels[2], 1)
        self.stage2_block4 = make_block(channels[2], channels[2], 1)
        self.stage2_block5 = make_block(channels[2], channels[2], 1)
        self.stage2_block6 = make_block(channels[2], channels[2], 1)
        self.stage2_block7 = make_block(channels[2], channels[2], 1)
        self.stage2_block8 = make_block(channels[2], channels[2], 1)

        self.stage3_block1 = make_block(channels[2], channels[3], 2)
        self.stage3_block2 = make_block(channels[3], channels[3], 1)
        self.stage3_block3 = make_block(channels[3], channels[3], 1)
        self.stage3_block4 = make_block(channels[3], channels[3], 1)

        self.conv_last = conv_bn_relu(channels[3], channels[4], kernel_size=1, stride=1, padding=0)
        self.globalpool = nn.AvgPool2d(input_size // 32)
        self.classifier = nn.Linear(channels[4], n_class)

        self._initialize_weights()


    def forward(self, x):
        x = self.conv_first(x)
        x = self.maxpool(x)

        x = self.stage1_block1(x)
        x = self.stage1_block2(x)
        x = self.stage1_block3(x)
        x = self.stage1_block4(x)

        x = self.stage2_block1(x)
        x = self.stage2_block2(x)
        x = self.stage2_block3(x)
        x = self.stage2_block4(x)
        x = self.stage2_block5(x)
        x = self.stage2_block6(x)
        x = self.stage2_block7(x)
        x = self.stage2_block8(x)

        x = self.stage3_block1(x)
        x = self.stage3_block2(x)
        x = self.stage3_block3(x)
        x = self.stage3_block4(x)

        x = self.conv_last(x)
        x = self.globalpool(x)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, LConv2d) and 'first' in name:
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LConv2d) and 'first' not in name:
                _, c, h, w = m.weight.shape
                nn.init.normal_(m.weight, 0, math.sqrt(1.0 / (c * h * w)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = ShuffleNetV2()
    print(model)
    flops, params = profile(model, input_size=(1,3,224,224), custom_ops={DRConv:count_Deformable_for_Dynamic_localshare_via_automask_gradient_Conv2d})
    print('flops = {} params = {}'.format(clever_format(flops), clever_format(params)))

