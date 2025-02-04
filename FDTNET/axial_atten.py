# -*- coding: utf-8 -*-
import torch.nn as nn
from conv_layer import Conv
from self_attention import self_attn


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = Conv(in_channel, out_channel, kSize=1,stride=1,padding=0)  #改变通道
        self.conv1 = Conv(out_channel, out_channel, kSize=(3, 3),stride = 1, padding=1)  #卷积操作
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)   #水平
        Wx = self.Wattn(Hx)   #垂直

        return Wx