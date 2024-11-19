import torch
from torch import nn
from torchvision import models as resnet_model
# import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Res2Net_v1b import res2net50_v1b_26w_4s
import numpy as np
from timm.models.layers import DropPath

# from thop import profile
# from thop import clever_format
# import torchvision.models as models
#
# from torchsummary import summary
#
# from torchstat import stat
#
# import time
#

from timm.models.layers import trunc_normal_
import math

__all__ = ['Pra_FATNet']

# 频谱的使用
class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, layer_id=None):
        super().__init__()
        if layer_id == 0:
            self.h = 24
            self.w = 13
        elif layer_id == 1:
            self.h = 48
            self.w = 25
        elif layer_id == 2:
            self.h = 96
            self.w = 49
        elif layer_id == 3:
            self.h = 12
            self.w = 7

        # self.complex_weight = nn.Parameter(torch.randn(dim, self.h, self.w, 2, dtype=torch.float32) * 0.02)

        self.dim = dim

    def forward(self, x):
        B, C, H, W = x.shape

        h = H
        w = h // 2 + 1
        complex_weight = nn.Parameter(torch.randn(self.dim, h, w, 2, dtype=torch.float32) * 0.02)

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

        weight = torch.view_as_complex(complex_weight)

        x = x.cuda()
        weight = weight.cuda()
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),  # kernel_size大小为1*3的窗口
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on. 密集聚合，可以被其他聚合替代
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x * y.expand_as(x)


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, padding=3, dilation=1, bias=False, act=True, stride=2):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True, stride=1):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class FAMBlock(nn.Module):
    def __init__(self, channels, layer_id, h):
        super(FAMBlock, self).__init__()


        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

        drop_path=0.
        self.norm = nn.LayerNorm(h)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.specter = SpectralGatingNetwork(dim=channels,layer_id=layer_id)


    def forward(self, x):
        x0 = x
        x0 = self.norm(x0)
        x0 = self.specter(x0)
        x0 = self.drop_path(x0)
        x = x + x0

        x3 = self.conv3(x)
        x3 = self.relu3(x3)

        x1 = self.conv1(x)
        x1 = self.relu1(x1)

        # x_specter = self.specter(self.norm(x))
        # x_specter = self.relu1(x_specter)

        out = x3 + x1


        return out

class decoderAttention(nn.Module):
    def __init__(self, in_channels,layer_id):
        super(decoderAttention, self).__init__()

        kernel_size = 3
        n_div = 4
        self.dim_conv = in_channels // n_div
        self.dim_untouched = in_channels - self.dim_conv * 3

        group_id = False
        if group_id:
            self.group = in_channels // n_div
        else:
            self.group = 1

        self.conv1 = nn.Conv2d(in_channels=self.dim_conv, out_channels=self.dim_conv, kernel_size=kernel_size,
                               stride=1, padding=(kernel_size - 1) // 2, groups=self.group)

        self.conv2_1 = nn.Conv2d(in_channels=self.dim_conv, out_channels=self.dim_conv, kernel_size=(1, 5),
                                 stride=1, padding=(0, 2), groups=self.group)
        self.conv2_2 = nn.Conv2d(in_channels=self.dim_conv, out_channels=self.dim_conv, kernel_size=(5, 1),
                                 stride=1, padding=(2, 0), groups=self.group)
        self.conv3_1 = nn.Conv2d(in_channels=self.dim_conv, out_channels=self.dim_conv, kernel_size=(1, 7),
                                 stride=1, padding=(0, 3), groups=self.group)
        self.conv3_2 = nn.Conv2d(in_channels=self.dim_conv, out_channels=self.dim_conv, kernel_size=(7, 1),
                                 stride=1, padding=(3, 0), groups=self.group)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        self.sgn= SpectralGatingNetwork(dim=self.dim_untouched,layer_id=layer_id)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv1(x1)
        x2 = self.conv2_2(self.conv2_1(x2))
        x3 = self.conv3_2(self.conv3_1(x3))
        x4 = self.sgn(x4)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv(x)
        return x


class Pra_FATNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=True):
        super(Pra_FATNet, self).__init__()
        channel = 32
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        self.affinity_attention = AffinityAttention(in_channels=2048)
        self.firstconv = self.resnet.conv1
        self.firstbn = self.resnet.bn1
        self.firstrelu = self.resnet.relu
        # self.cb = Conv2D()

        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_384', pretrained=True)
        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )
        self.conv_seq_img = nn.Conv2d(in_channels=3072, out_channels=2048,kernel_size=1, padding=0)
        self.se = squeeze_excitation_block(in_channels=4096)
        self.conv2d = nn.Conv2d(in_channels=4096, out_channels=2048, kernel_size=1, padding=0)
        self.cb = ConvBlock(512, 32)
        self.cb2 = ConvBlock(32, 1)

        self.FAMBlock1 = FAMBlock(channels=256,layer_id=2,h=96)
        self.FAMBlock2 = FAMBlock(channels=512,layer_id=1,h=48)
        self.FAMBlock3 = FAMBlock(channels=1024,layer_id=0,h=24)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [256, 512, 1024, 2048]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_decoder1 = DecoderBottleneckLayer(256,128)
        self.final_decoder2 = DecoderBottleneckLayer(128,64)

        self.final_conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, 1, 3, padding=1)

        self.featuredecoderAttention = decoderAttention(in_channels=2048, layer_id=3)
        self.decoderAttention4 = decoderAttention(in_channels=1024,layer_id=0)
        self.decoderAttention3 = decoderAttention(in_channels=512,layer_id=1)






        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)  # 聚合
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        y = x

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.resnet.maxpool(x)
        print('x',x.shape)

        # ---- low-level features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        print('x1', x1.shape)
        print('x2', x2.shape)
        print('x3', x3.shape)
        print('x4', x4.shape)

        # ---- RFB ----
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)  # channel -> 32
        x4_rfb = self.rfb4_1(x4)  # channel -> 32

        # x2_attention = self.affinity_attention(x2_rfb)
        # x3_attention = self.affinity_attention(x3_rfb)
        # x4_attention = self.affinity_attention(x4_rfb)
        #
        # x2_attention_fuse = x2_attention + x2_rfb
        # x3_attention_fuse = x3_attention + x3_rfb
        # x4_attention_fuse = x4_attention + x4_rfb



        emb = self.patch_embed(y)

        for i in range(12):
            emb = self.transformers[i](emb)

        print('emb',emb.shape)

        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 3072, 12, 12)
        feature_tf = self.conv_seq_img(feature_tf)
        feature_cat = torch.cat((x4, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)
        # feature_out = self.cb(feature_out)



        for i in range(2):
            e3 = self.FAM3[i](x3)
        for i in range(4):
            e2 = self.FAM2[i](x2)
        for i in range(6):
            e1 = self.FAM1[i](x1)

        print('e3', e3.shape)
        print('e2', e2.shape)
        print('e1', e1.shape)

        feature_out_attention = self.affinity_attention(feature_out)
        feature_out = feature_out_attention + feature_out

        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        print('feature_out', feature_out.shape)
        print('d4', d4.shape)
        print('d3', d3.shape)
        print('d2', d2.shape)


        feature_out = self.featuredecoderAttention(feature_out)
        d4 = self.decoderAttention4(d4)
        d3 = self.decoderAttention3(d3)




        feature_out = self.ra4_conv1(feature_out)
        feature_out = F.relu(self.ra4_conv2(feature_out))
        feature_out = F.relu(self.ra4_conv3(feature_out))
        feature_out = F.relu(self.ra4_conv4(feature_out))
        feature_out = self.ra4_conv5(feature_out)

        d4 = self.ra3_conv1(d4)
        d4 = F.relu(self.ra3_conv2(d4))
        d4 = F.relu(self.ra3_conv3(d4))
        d4 = self.ra3_conv4(d4)

        d3 = self.ra2_conv1(d3)
        d3 = F.relu(self.ra2_conv2(d3))
        d3 = F.relu(self.ra2_conv3(d3))
        d3 = self.ra2_conv4(d3)


        # x1->d2
        # x2->d3
        # x3->d4
        # x4->feature_out



        # ---- Partial Decoder ----    PD模块
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        print('ra5_feat',ra5_feat.shape)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        print('ra4_feat',ra4_feat.shape)
        x = ra4_feat + crop_4 + feature_out
        lateral_map_4 = F.interpolate(x, scale_factor=32,
                                      mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        print('ra3_feat', ra3_feat.shape)
        x = ra3_feat + crop_3 + d4
        lateral_map_3 = F.interpolate(x, scale_factor=16,
                                      mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        # print('lateral_map_3',lateral_map_3.shape)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        print('ra2_feat', ra2_feat.shape)
        print('d3',d3.shape)
        x = ra2_feat + crop_2 + d3
        lateral_map_2 = F.interpolate(x, scale_factor=8,
                                      mode='bilinear')  # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # print('lateral_map_2', lateral_map_2.shape)


        return [lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2]


if __name__ == '__main__':
    # device_ids = [0, 1]  # 2 gpu
    #
    # images = torch.rand(1, 3, 224, 224).cuda(device=device_ids[0])
    # model = Pranet(num_classes=1,input_channels=3,deep_supervision=True)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(device=device_ids[0])

    images = torch.rand(1, 3, 384, 384).cuda(0)
    model = Pra_FATNet(num_classes=1, input_channels=3, deep_supervision=True)
    model = model.cuda(0)

    outputs = model(images)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    print(outputs[3].shape)

    # images = torch.rand(1, 3, 224, 224).cuda(0)
    # model = FATNet(num_classes=1,input_channels=3,deep_supervision=True)
    # # model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda(0)
    # output = model(images)
    # print('output', output.shape)

    # model = FATNet(num_classes=1,input_channels=3,deep_supervision=True)
    # input = torch.rand((1, 3, 224, 224)).cuda(0)
    # model = model.cuda(0)
    # flops, params = profile(model, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    #
    # print("Params：", params)
    # print("GFLOPS：", flops)

    # model = FATNet(num_classes=1,input_channels=3,deep_supervision=True).eval().cuda()
    # summary(model, input_size=(3, 224, 224), batch_size=-1)

    # model = FATNet(num_classes=1, input_channels=3, deep_supervision=True)
    # print(stat(model, (3, 224, 224)))

    # # 进行模型推理，并计算FPS
    # num_iterations = 500  # 进行100次推理来计算平均FPS
    # total_time = 0
    #
    # with torch.no_grad():
    #     for _ in range(num_iterations):
    #         start_time = time.time()
    #         output = model(input)
    #         end_time = time.time()
    #         total_time += end_time - start_time
    #
    # average_fps = num_iterations / total_time
    # print(f"Average FPS: {average_fps}")

