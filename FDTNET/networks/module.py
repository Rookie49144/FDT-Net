from swin_transformer_v2 import SwinTransformerV2
import torch
import math
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, channels, depth=4):
        super().__init__()
        self.align = nn.ModuleList()
        for i in range(depth - 1):
            self.align.append(nn.Sequential(nn.Conv2d(channels * (2 ** (i + 1)), channels * (2 ** i), 1),
                                            nn.UpsamplingBilinear2d(scale_factor=2)))

    def forward(self, feature_list):
        main_feature = feature_list[-1]
        for i in range(len(feature_list) - 2, -1, -1):
            main_feature = self.align[i](main_feature) + feature_list[i]
        return main_feature


class Head(nn.Module):
    def __init__(self, in_channels=512, mid_channels=64, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(mid_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                   nn.Conv2d(mid_channels, num_classes, kernel_size=1))

    def forward(self, feature):
        feature = self.conv1(feature)
        out = self.conv2(feature)
        return out


class SegSwinTransformer(SwinTransformerV2):
    def __init__(self, img_size=128, window_size=4, embed_dim=128, num_class=2):
        super().__init__(img_size=img_size, window_size=window_size, embed_dim=embed_dim,
                         num_heads=[4, 8, 16, 32])
        self.num_classes = num_class
        self.decoder = Decoder(embed_dim)
        self.seg_head = Head(embed_dim)

    def forward_features(self, x):
        output_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.layers)):
            shape = [x.shape[0], x.shape[2], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))]
            output_list.append(torch.transpose(x, -1, -2).reshape(shape))
            x = self.layers[i](x)
        return output_list

    def forward(self, x):
        x = torch.cat([x, x, x], dim=1)
        x = self.forward_features(x)
        x = self.decoder(x)
        x = self.seg_head(x)
        return torch.softmax(x, dim=1)


if __name__ == "__main__":
    model = SegSwinTransformer()
    x = torch.rand([3, 1, 128, 128])
    y = model(x)
    print(y.shape)
