# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from conv_layer import Conv


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))   #一个可训练的参数，用于缩放自注意力计算的结果
        self.softmax = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).contiguous().view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).contiguous().view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).contiguous().view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.contiguous().view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out

"""if __name__ == '__main__':
    x = torch.rand(2, 32, 16, 16)
    model = self_attn(32,'h')
    y = model(x)"""