# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import cv2

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import seg_configs as configs
from .seg_modeling_resnet_skip import ResNetV2
from axial_atten import AA_kernel
from conv_layer import Conv

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)

        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.dwconv = DepthWiseConv(768,768)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer1 = self.transpose_for_scores(mixed_query_layer)
        key_layer1 = self.transpose_for_scores(mixed_key_layer)
        value_layer1 = self.transpose_for_scores(mixed_value_layer)
        #global
        avgpool = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        channel = query_layer1.shape[1]
        conv = nn.Conv2d(channel, channel, kernel_size=3, stride=4, padding=1)
        q = avgpool(query_layer1)
        k = avgpool(key_layer1)
        v = avgpool(value_layer1)
        atten = torch.matmul(q, k.transpose(-1, -2))
        atten = atten / math.sqrt(self.attention_head_size)
        atten = self.softmax(atten)
        w = atten if self.vis else None
        atten = self.attn_dropout(atten)
        cont = torch.matmul(atten, v)
        cont = F.interpolate(cont, scale_factor=4, mode='bilinear', align_corners=False)
        #local
        mq = mixed_query_layer.permute(0,2,1)
        mQ = mq.view(2,768,32,32)
        Q = self.dwconv(mQ)
        mk = mixed_key_layer.permute(0, 2, 1)
        mK = mk.view(2, 768, 32, 32)
        K = self.dwconv(mK)
        mv = mixed_value_layer.permute(0, 2, 1)
        mV = mv.view(2, 768, 32, 32)
        V = self.dwconv(mV)
        Q = Q.flatten(2)
        mixed_query_layer = Q.transpose(-1, -2)
        K = K.flatten(2)
        mixed_key_layer = K.transpose(-1, -2)
        V = V.flatten(2)
        mixed_value_layer = V.transpose(-1, -2)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.mul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.mul(attention_probs, value_layer)
        #concatenation
        concatenated = torch.cat((context_layer, cont), dim=1)
        conv_1x1 = nn.Conv2d(concatenated.shape[1], context_layer.shape[1], kernel_size=1).cuda()
        context_layer = conv_1x1(concatenated)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        #Discrete Cosine Transform(DCT)
        for i in range(x.shape[0]):
            x_np = x[i].detach().cpu().numpy()
            x_np = x_np.astype(np.float32)
            channels = cv2.split(x_np)
            dct_channels = [cv2.dct(channel) for channel in channels]
            dct_image = cv2.merge(dct_channels)
            threshold = 0.01
            dct_image = dct_image * (np.abs(dct_image) > threshold)
            x[i] = torch.tensor(dct_image,requires_grad=True).cuda()
        x, weights = self.attn(x)
        #IDCT
        for i in range(x.shape[0]):
            x_np = x[i].detach().cpu().numpy()
            x_np = x_np.astype(np.float32)
            channels = cv2.split(x_np)
            dct_channels = [cv2.idct(channel) for channel in channels]
            dct_image = cv2.merge(dct_channels)
            x[i] = torch.tensor(dct_image,requires_grad=True).cuda()
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        a = x
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x,a
class Ed(nn.Module):
    def __init__(self):
        '''----- PCIM -----'''
        super(Ed, self).__init__()
        self.to_channel0 = nn.Conv2d(3, 64, (1, 1), (1, 1), 0)
        self.to_channel1 = nn.Conv2d(3, 256, (1, 1), (1, 1), 0)
        self.to_channel2 = nn.Conv2d(3, 512, (1, 1), (1, 1), 0)
        ''' Axial Attention'''
        self.aa_kernel_1 = AA_kernel(64, 64)
        self.aa_kernel_2 = AA_kernel(256, 256)
        self.aa_kernel_3 = AA_kernel(512, 512)
        '''RA part'''
        self.ra1_conv1 = Conv(64, 64, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv2 = Conv(64, 64, 3, 1, padding=1, bn_acti=True)
        self.ra1_conv3 = Conv(64, 3, 3, 1, padding=1, bn_acti=True)

        self.ra2_conv1 = Conv(256, 256, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv2 = Conv(256, 256, 3, 1, padding=1, bn_acti=True)
        self.ra2_conv3 = Conv(256, 3, 3, 1, padding=1, bn_acti=True)

        self.ra3_conv1 = Conv(512, 512, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv2 = Conv(512, 512, 3, 1, padding=1, bn_acti=True)
        self.ra3_conv3 = Conv(512, 3, 3, 1, padding=1, bn_acti=True)
        '''Feature Concat'''
        self.concat1 = Conv(128, 64, 1, 1, 0, bn_acti=True)
        self.concat2 = Conv(512, 256, 1, 1, 0, bn_acti=True)
        self.concat3 = Conv(1024, 512, 1, 1, 0, bn_acti=True)
        self.cor_conv1_1 = Conv(64, 64, 3, 1, 1, bn_acti=True)
        self.cor_conv1_2 = Conv(64, 64, 3, 1, 1, bn_acti=True)
        self.cor_conv2_1 = Conv(256, 256, 3, 1, 1, bn_acti=True)
        self.cor_conv2_2 = Conv(256, 256, 3, 1, 1, bn_acti=True)
        self.cor_conv3_1 = Conv(512, 512, 3, 1, 1, bn_acti=True)
        self.cor_conv3_2 = Conv(512, 512, 3, 1, 1, bn_acti=True)
    def forward(self,Feature,y):
        f0 = Feature[2]
        f1 = Feature[1]
        f2 = Feature[0]
        aa_atten_2 = self.aa_kernel_3(f2)  # feature map
        y0 = F.interpolate(y, size=(64, 64), mode='bilinear', align_corners=True)
        y_soft2 = torch.softmax(y0, dim=1)
        y_mask2 = self.to_channel2(y_soft2)  # mask
        aa_atten_2_1 = y_mask2.mul(aa_atten_2)  # feature * mask
        m_decider_2_ra = 1 - y_mask2  # 1 - mask
        aa_atten_2_2 = m_decider_2_ra.mul(aa_atten_2)  # feature * (1-mask)

        aa_atten_2_11 = self.cor_conv3_1(aa_atten_2_1)
        aa_atten_2_22 = self.cor_conv3_2(aa_atten_2_2)
        cor_1 = aa_atten_2_11.mul(aa_atten_2_22)  # correlation
        aa_atten_2_o = self.concat3(torch.cat([cor_1, aa_atten_2_11], dim=1))

        ra_2 = self.ra3_conv1(aa_atten_2_o)  # 512 --> 512
        ra_2 = self.ra3_conv2(ra_2)  # 512 --> 512
        ra_2 = self.ra3_conv3(ra_2)  # 512 --> 3

        """________"""
        aa_atten_1 = self.aa_kernel_2(f1)  # feature map
        y1 = F.interpolate(y, size=(128, 128), mode='bilinear', align_corners=True)
        y_soft1 = torch.softmax(y1, dim=1)
        y_mask1 = self.to_channel1(y_soft1)  # mask
        aa_atten_1_1 = y_mask1.mul(aa_atten_1)  # feature * mask
        m_decider_1_ra = 1 - y_mask1  # 1 - mask
        aa_atten_1_2 = m_decider_1_ra.mul(aa_atten_1)  # feature * (1-mask)

        aa_atten_1_11 = self.cor_conv2_1(aa_atten_1_1)
        aa_atten_1_22 = self.cor_conv2_2(aa_atten_1_2)
        cor_1 = aa_atten_1_11.mul(aa_atten_1_22)  # correlation
        aa_atten_1_o = self.concat2(torch.cat([cor_1, aa_atten_1_11], dim=1))

        ra_1 = self.ra2_conv1(aa_atten_1_o)  # 256 --> 256
        ra_1 = self.ra2_conv2(ra_1)  # 256 --> 256
        ra_1 = self.ra2_conv3(ra_1)  # 256 --> 3

        """__________"""
        aa_atten_0 = self.aa_kernel_1(f0)  # feature map
        y2 = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=True)
        y_soft0 = torch.softmax(y2, dim=1)
        y_mask0 = self.to_channel0(y_soft0)  # mask
        aa_atten_0_1 = y_mask0.mul(aa_atten_0)  # feature * mask
        m_decider_0_ra = 1 - y_mask0  # 1 - mask
        aa_atten_0_2 = m_decider_0_ra.mul(aa_atten_0)  # feature * (1-mask)
        aa_atten_0_11 = self.cor_conv1_1(aa_atten_0_1)
        aa_atten_0_22 = self.cor_conv1_2(aa_atten_0_2)
        cor_0 = aa_atten_0_11.mul(aa_atten_0_22)  # correlation
        aa_atten_0_o = self.concat1(torch.cat([cor_0, aa_atten_0_11], dim=1))

        ra_0 = self.ra1_conv1(aa_atten_0_o)  # 128 --> 128
        ra_0 = self.ra1_conv2(ra_0)  # 128 --> 128
        ra_0 = self.ra1_conv3(ra_0)  # 128 --> 3

        #x_0 = ra_0 + x_1
        ra_0 = F.interpolate(ra_0, size=(512,512), mode='bilinear', align_corners=True)
        ra_1 = F.interpolate(ra_1, size=(512,512), mode='bilinear', align_corners=True)
        ra_2 = F.interpolate(ra_2, size=(512,512), mode='bilinear', align_corners=True)
        x_0 = ra_0 + ra_1 + ra_2 + y
        lateral_map_0 = x_0
        return lateral_map_0


class SegModel(nn.Module):
    def __init__(self, config, img_size=512, num_classes=3, zero_head=False, vis=False):
        super(SegModel, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.ed = Ed()
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x,a = self.decoder(x, features)
        logits = self.segmentation_head(x)
        out = self.ed(features,logits)
        return out

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'Seg_model': configs.get_seg_config(),
    'testing': configs.get_testing(),
}


