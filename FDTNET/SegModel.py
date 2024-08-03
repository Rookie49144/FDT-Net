import torch
from networks.seg_modeling import SegModel as Segmodel
from networks.seg_modeling import CONFIGS as CONFIGS_seg
import torch.nn as nn
import torch.nn.functional as F


class FDTNet(nn.Module):
    def __init__(self):
        super().__init__()
        name = 'Seg_model'
        num_classes = 3
        n_skip = 3
        img_size = 512
        patches_size = 16

        config_seg = CONFIGS_seg[name]
        config_seg.n_classes = num_classes
        config_seg.n_skip = n_skip
        config_seg.patches.grid = (int(img_size / patches_size), int(img_size / patches_size))

        self.net = Segmodel(config_seg, img_size=img_size, num_classes=config_seg.n_classes)


    def forward(self, img):
        out = self.net(img)
        evidence = self.infer(out)  #softplus

        return evidence
    def infer(self, input):
        evidence = F.softplus(input)
        return evidence

