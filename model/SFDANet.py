import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt.data
from einops import rearrange
import numbers
import copy
from thop import profile, clever_format
from torch.nn.modules.utils import _pair, _triple
from torch_dct import dct_2d, idct_2d
from pytorch_wavelets import DWTForward
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
import numpy as np

den = [0.5]


def carafe(x, normed_mask, kernel_size, group=1, up=1):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    # assert m_c == kernel_size ** 2 * up ** 2
    assert m_h == up * h
    assert m_w == up * w
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1,
                        padding=0)  # F.unfold 是用于从输入特征图中提取滑动窗口（局部区域）的函数
    # unfold_x = unfold_x.reshape(b, c, 1, kernel_size, kernel_size, h, w).repeat(1, 1, up ** 2, 1, 1, 1, 1)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
    # normed_mask = normed_mask.reshape(b, 1, up ** 2, kernel_size, kernel_size, h, w)
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    # test
    # res[:, :, 0] = 1
    # res[:, :, 1] = 2
    # res[:, :, 2] = 3
    # res[:, :, 3] = 4
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    # res = F.pixel_shuffle(res, up)
    return res


class wConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, den, stride=1, padding=1, groups=1, dilation=1,
                 bias=False):
        super().__init__()
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size = _pair(kernel_size)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        device = torch.device('cpu')
        self.register_buffer('alfa', torch.cat([torch.tensor(den, device=device), torch.tensor([1.0], device=device),
                                                torch.flip(torch.tensor(den, device=device), dims=[0])]))
        self.register_buffer('Phi', torch.outer(self.alfa, self.alfa))

        if self.Phi.shape != self.kernel_size:
            raise ValueError(f"Phi shape {self.Phi.shape} must match kernel size {self.kernel_size}")

    def forward(self, x):
        Phi = self.Phi.to(x.device)
        weight_Phi = self.weight * Phi
        return F.conv2d(x, weight_Phi, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups,
                        dilation=self.dilation)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, _ = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    # x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c // 4, padding=pad)
    return x



class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale, requires_grad=True)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CrsSpaAttn(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], patchSize=[32, 16, 8, 4], layer_num=1, size=256):
        super().__init__()
        patchSize = [size // 8, size // 16, size // 32, size // 64]  # patch32(256), 64(512)
        overlap = [p // 2 for p in patchSize]  # p/2
        self.embeddings_1 = AvgMax_Embeddings()
        self.embeddings_2 = AvgMax_Embeddings()
        self.embeddings_3 = AvgMax_Embeddings()
        self.embeddings_4 = AvgMax_Embeddings()

        # self.embeddings_1 = Channel_Embeddings(patchSize[0], in_channels=channel_num[0], overlap=overlap[0])
        # self.embeddings_2 = Channel_Embeddings(patchSize[1], in_channels=channel_num[1], overlap=overlap[1])
        # self.embeddings_3 = Channel_Embeddings(patchSize[2], in_channels=channel_num[2], overlap=overlap[2])
        # self.embeddings_4 = Channel_Embeddings(patchSize[3], in_channels=channel_num[3], overlap=overlap[3])

        self.layer = nn.ModuleList()
        for _ in range(layer_num):
            layer = MLA_Block(channel_num, mode='kernel')
            self.layer.append(copy.deepcopy(layer))

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1)
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1)
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1)
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1)

    def forward(self, en1, en2, en3, en4):
        # emb1 = self.embeddings_1(en1)
        # emb2 = self.embeddings_2(en2)
        # emb3 = self.embeddings_3(en3)
        # emb4 = self.embeddings_4(en4)

        for layer_block in self.layer:
            en1, en2, en3, en4 = layer_block(en1, en2, en3, en4)

        # org_size = [(en1.shape[2], en1.shape[3]), (en2.shape[2], en2.shape[3]), (en3.shape[2], en3.shape[3]),
        #             (en4.shape[2], en4.shape[3])]
        # emb1 = self.reconstruct_1(emb1, size=org_size[0]) if emb1 is not None else None
        # emb2 = self.reconstruct_2(emb2, size=org_size[1]) if emb2 is not None else None
        # emb3 = self.reconstruct_3(emb3, size=org_size[2]) if emb3 is not None else None
        # emb4 = self.reconstruct_4(emb4, size=org_size[3]) if emb4 is not None else None
        #
        # out1 = emb1 + en1 if en1 is not None else None
        # out2 = emb2 + en2 if en2 is not None else None
        # out3 = emb3 + en3 if en3 is not None else None
        # out4 = emb4 + en4 if en4 is not None else None

        return en1, en2, en3, en4


class MLA_Block(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel'):
        super().__init__()
        self.msa = MSA(channel_num)  # sep or sep_gate or same_conv # 修改这里
        self.mca = MCA(channel_num, mode=mode)
        self.CrsSpaAttnBlock = CrsSpaAttnBlock(channel_num, mode=mode)
        self.mca_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.mca_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.mca_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.mca_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        size = 128
        patchSize = [size // 8, size // 16, size // 32, size // 64]  # patch32(256), 64(512)
        overlap = [0,0,0,0]
        self.embeddings_1 = Channel_Embeddings(patchSize[0], in_channels=channel_num[0], overlap=overlap[0])
        self.embeddings_2 = Channel_Embeddings(patchSize[1], in_channels=channel_num[1], overlap=overlap[1])
        self.embeddings_3 = Channel_Embeddings(patchSize[2], in_channels=channel_num[2], overlap=overlap[2])
        self.embeddings_4 = Channel_Embeddings(patchSize[3], in_channels=channel_num[3], overlap=overlap[3])

        # self.embeddings_1 = AvgMax_Embeddings()
        # self.embeddings_2 = AvgMax_Embeddings()
        # self.embeddings_3 = AvgMax_Embeddings()
        # self.embeddings_4 = AvgMax_Embeddings()

        self.proj1 = nn.Linear(channel_num[0], channel_num[0])
        self.proj2 = nn.Linear(channel_num[1], channel_num[1])
        self.proj3 = nn.Linear(channel_num[2], channel_num[2])
        self.proj4 = nn.Linear(channel_num[3], channel_num[3])

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1)
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1)
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1)
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1)

    def forward(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)
        # res1, res2, res3, res4 = emb1, emb2, emb3, emb4
        sa1, sa2, sa3, sa4 = self.CrsSpaAttnBlock(emb1, emb2, emb3, emb4)
        # ca1, ca2, ca3, ca4 = self.mca(emb1, emb2, emb3, emb4)
        # ca1 = ca1 + res1
        # ca2 = ca2 + res2
        # ca3 = ca3 + res3
        # ca4 = ca4 + res4
        b,c1,h,w = sa1.shape
        b,c2,_,_ = sa2.shape
        b,c3,_,_ = sa3.shape
        b,c4,_,_ = sa4.shape
        sa1 = self.proj1(sa1.reshape(b, c1, -1).transpose(1, 2)).transpose(1, 2).reshape(b, c1, h, w)
        sa2 = self.proj2(sa2.reshape(b, c2, -1).transpose(1, 2)).transpose(1, 2).reshape(b, c2, h, w)
        sa3 = self.proj3(sa3.reshape(b, c3, -1).transpose(1, 2)).transpose(1, 2).reshape(b, c3, h, w)
        sa4 = self.proj4(sa4.reshape(b, c4, -1).transpose(1, 2)).transpose(1, 2).reshape(b, c4, h, w)

        org_size = [(en1.shape[2], en1.shape[3]), (en2.shape[2], en2.shape[3]), (en3.shape[2], en3.shape[3]),
                    (en4.shape[2], en4.shape[3])]

        # sa1 = F.interpolate(sa1, size=org_size[0], mode='bilinear', align_corners=False)
        # sa2 = F.interpolate(sa2, size=org_size[1], mode='bilinear', align_corners=False)
        # sa3 = F.interpolate(sa3, size=org_size[2], mode='bilinear', align_corners=False)
        # sa4 = F.interpolate(sa4, size=org_size[3], mode='bilinear', align_corners=False)


        # org_size = [(en1.shape[2], en1.shape[3]), (en2.shape[2], en2.shape[3]), (en3.shape[2], en3.shape[3]),
        #             (en4.shape[2], en4.shape[3])]
        #
        sa1 = self.reconstruct_1(sa1, size=org_size[0])
        sa2 = self.reconstruct_2(sa2, size=org_size[1])
        sa3 = self.reconstruct_3(sa3, size=org_size[2])
        sa4 = self.reconstruct_4(sa4, size=org_size[3])

        # ca1 = self.mca_norm1(ca1) if ca1 is not None else None
        # ca2 = self.mca_norm2(ca2) if ca2 is not None else None
        # ca3 = self.mca_norm3(ca3) if ca3 is not None else None
        # ca4 = self.mca_norm4(ca4) if ca4 is not None else None

        # return ca1, ca2, ca3, ca4
        # sa1, sa2, sa3, sa4 = self.mca(ca1, ca2, ca3, ca4)
        # sa1, sa2, sa3, sa4, sa_weight = self.msa(ca1, ca2, ca3, ca4)

        sa1 = sa1 + en1
        sa2 = sa2 + en2
        sa3 = sa3 + en3
        sa4 = sa4 + en4

        return sa1, sa2, sa3, sa4


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, size=None):
        if x is None:
            return None

        if size is not None:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Channel_Embeddings(nn.Module):
    def __init__(self, patchsize, in_channels, overlap):
        super().__init__()
        # patch_size = _pair(patchsize)
        # n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # 14 * 14 = 196

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patchsize,
                                          stride=patchsize - overlap)
        # self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        # self.dropout = nn.Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # b, c, 31, 31
        return x

class AvgMax_Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        # self.patch_embeddings = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=patchsize,
        #                                   stride=patchsize - overlap)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, (16, 16))
        max = F.adaptive_max_pool2d(x, (16, 16))
        return avg + max


class MSA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128]):  # sum[16, 32, 64, 128, 256]
        super().__init__()
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]  # 240
        self.conv_emb = nn.Conv2d(self.channel_size, self.channel_size // 8, 1, 1, padding='same')
        self.emb_norm = nn.BatchNorm2d(self.channel_size // 8)
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(channel_num[3], channel_num[3] // 8, 1, 1, padding='same'),  # channel_num[3] // 8
            nn.BatchNorm2d(channel_num[3] // 8),
            nn.ReLU()
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(channel_num[2], channel_num[2], 3, 1, padding='same', groups=channel_num[2]),
            nn.Conv2d(channel_num[2], channel_num[2] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[2] // 8),
            nn.ReLU()
        )
        self.Conv_5 = nn.Sequential(
            nn.Conv2d(channel_num[1], channel_num[1], 5, 1, padding='same', groups=channel_num[1]),
            nn.Conv2d(channel_num[1], channel_num[1] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[1] // 8),
            nn.ReLU()
        )
        self.Conv_7 = nn.Sequential(
            nn.Conv2d(channel_num[0], channel_num[0], 7, 1, padding='same', groups=channel_num[0]),
            nn.Conv2d(channel_num[0], channel_num[0] // 8, 1, 1, padding='same'),
            nn.BatchNorm2d(channel_num[0] // 8),
            nn.ReLU()
        )
        self.se = SE_Block(self.channel_size // 8 * 2, ratio=10)  # 5, 10
        self.final_Conv = nn.Conv2d(self.channel_size // 8 * 2, 1, 1, 1, padding='same')

        self.sig = nn.Sigmoid()

    def forward(self, emb1, emb2, emb3, emb4):
        emb = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        emb = self.conv_emb(emb)
        emb = self.emb_norm(emb)
        emb_1 = self.Conv_7(emb1)
        emb_3 = self.Conv_5(emb2)
        emb_5 = self.Conv_3(emb3)
        emb_7 = self.Conv_1(emb4)

        sa = self.final_Conv(self.se(torch.cat([emb_1, emb_3, emb_5, emb_7, emb], dim=1)))
        sa = self.sig(sa)

        emb_1 = sa * emb1
        emb_2 = sa * emb2
        emb_3 = sa * emb3
        emb_4 = sa * emb4

        emb_1 = emb_1 + emb1
        emb_2 = emb_2 + emb2
        emb_3 = emb_3 + emb3
        emb_4 = emb_4 + emb4

        return emb_1, emb_2, emb_3, emb_4, sa


class MCA(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel'):
        super().__init__()
        self.mode = mode
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(self.channel_size, LayerNorm_type='WithBias')

        if self.mode == 'kernel':
            self.q1_Conv = nn.Conv2d(channel_num[0], channel_num[0], 1, 1)
            self.k1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)
            self.v1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)

            self.q2_Conv = nn.Conv2d(channel_num[1], channel_num[1], 1, 1)
            self.k2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)
            self.v2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)

            self.q3_Conv = nn.Conv2d(channel_num[2], channel_num[2], 1, 1)
            self.k3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)
            self.v3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)

            self.q4_Conv = nn.Conv2d(channel_num[3], channel_num[3], 1, 1)
            self.k4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)
            self.v4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)

            self.out_Conv1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
            self.out_Conv2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
            self.out_Conv3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
            self.out_Conv4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

            self.softmax = nn.Softmax(dim=1)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)

    def kernel_similarity(self, Gq, Gk, sigma=None):
        assert Gq.shape == Gk.shape

        diff = Gq - Gk
        if sigma is None:
            sigma = torch.std(diff, dim=2).unsqueeze(-1)
        distance = torch.norm(diff, p=2, dim=2).unsqueeze(-1)
        similarity = torch.exp(-distance ** 2 / (2 * sigma ** 2))  # GUSSIAN KERNEL

        return similarity

    def forward(self, emb1, emb2, emb3, emb4):
        b, c, h, w = emb1.shape
        emb_all = torch.cat([emb1, emb2, emb3, emb4], dim=1)
        emb1_norm = self.attn_norm1(emb1) if emb1 is not None else None
        emb2_norm = self.attn_norm2(emb2) if emb2 is not None else None
        emb3_norm = self.attn_norm3(emb3) if emb3 is not None else None
        emb4_norm = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)

        if self.mode == 'kernel':
            emb1_gq = self.q1_Conv(emb1_norm)
            emb_all_gk1 = self.k1_Conv(emb_all)
            emb_all_gv1 = self.v1_Conv(emb_all)
            emb1_gq = rearrange(emb1_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk1 = rearrange(emb_all_gk1, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv1 = rearrange(emb_all_gv1, 'b c h w -> b c (h w)', h=h, w=w)
            emb1_gq, emb_all_gk1 = (F.normalize(emb1_gq, dim=-1), F.normalize(emb_all_gk1, dim=-1))

            emb2_gq = self.q2_Conv(emb2_norm)
            emb_all_gk2 = self.k2_Conv(emb_all)
            emb_all_gv2 = self.v2_Conv(emb_all)
            emb2_gq = rearrange(emb2_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk2 = rearrange(emb_all_gk2, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv2 = rearrange(emb_all_gv2, 'b c h w -> b c (h w)', h=h, w=w)
            emb2_gq, emb_all_gk2 = (F.normalize(emb2_gq, dim=-1), F.normalize(emb_all_gk2, dim=-1))

            emb3_gq = self.q3_Conv(emb3_norm)
            emb_all_gk3 = self.k3_Conv(emb_all)
            emb_all_gv3 = self.v3_Conv(emb_all)
            emb3_gq = rearrange(emb3_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk3 = rearrange(emb_all_gk3, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv3 = rearrange(emb_all_gv3, 'b c h w -> b c (h w)', h=h, w=w)
            emb3_gq, emb_all_gk3 = (F.normalize(emb3_gq, dim=-1), F.normalize(emb_all_gk3, dim=-1))

            emb4_gq = self.q4_Conv(emb4_norm)
            emb_all_gk4 = self.k4_Conv(emb_all)
            emb_all_gv4 = self.v4_Conv(emb_all)
            emb4_gq = rearrange(emb4_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk4 = rearrange(emb_all_gk4, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv4 = rearrange(emb_all_gv4, 'b c h w -> b c (h w)', h=h, w=w)
            emb4_gq, emb_all_gk4 = (F.normalize(emb4_gq, dim=-1), F.normalize(emb_all_gk4, dim=-1))

            sim1 = self.kernel_similarity(emb1_gq, emb_all_gk1)
            sim2 = self.kernel_similarity(emb2_gq, emb_all_gk2)
            sim3 = self.kernel_similarity(emb3_gq, emb_all_gk3)
            sim4 = self.kernel_similarity(emb4_gq, emb_all_gk4)

            sim1, sim2, sim3, sim4 = self.softmax(sim1), self.softmax(sim2), self.softmax(sim3), self.softmax(sim4)

            out1 = sim1 * emb_all_gv1
            out2 = sim2 * emb_all_gv2
            out3 = sim3 * emb_all_gv3
            out4 = sim4 * emb_all_gv4

            out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h, w=w)
            out2 = rearrange(out2, 'b c (h w) -> b c h w', h=h, w=w)
            out3 = rearrange(out3, 'b c (h w) -> b c h w', h=h, w=w)
            out4 = rearrange(out4, 'b c (h w) -> b c h w', h=h, w=w)

            att_out1 = self.out_Conv1(out1)
            att_out2 = self.out_Conv2(out2)
            att_out3 = self.out_Conv3(out3)
            att_out4 = self.out_Conv4(out4)

            att_out1 = att_out1 + emb1 if emb1 is not None else None
            att_out2 = att_out2 + emb2 if emb2 is not None else None
            att_out3 = att_out3 + emb3 if emb3 is not None else None
            att_out4 = att_out4 + emb4 if emb4 is not None else None

            res1 = att_out1
            res2 = att_out2
            res3 = att_out3
            res4 = att_out4
            x1 = self.ffn_norm1(res1) if emb1 is not None else None
            x2 = self.ffn_norm2(res2) if emb2 is not None else None
            x3 = self.ffn_norm3(res3) if emb3 is not None else None
            x4 = self.ffn_norm4(res4) if emb4 is not None else None
            x1 = self.ffn1(x1) if emb1 is not None else None
            x2 = self.ffn2(x2) if emb2 is not None else None
            x3 = self.ffn3(x3) if emb3 is not None else None
            x4 = self.ffn4(x4) if emb4 is not None else None
            x1 = x1 + res1 if emb1 is not None else None
            x2 = x2 + res2 if emb2 is not None else None
            x3 = x3 + res3 if emb3 is not None else None
            x4 = x4 + res4 if emb4 is not None else None

            return x1, x2, x3, x4


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.GELU,
                 extra_act=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.act1 = act_layer() if extra_act else nn.Identity()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize, padding=ksize // 2, stride=1,
                              groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.conv(x)
        x = self.fc2(x)
        return x


class CrsSpaAttnBlock(nn.Module):
    def __init__(self, channel_num=[16, 32, 64, 128], mode='kernel', base_pooled_sizes=[11, 8, 6, 4], num_heads=2):
        super().__init__()
        self.mode = mode
        self.num_heads = num_heads
        self.base_pooled_sizes = base_pooled_sizes
        self.channel_size = channel_num[0] + channel_num[1] + channel_num[2] + channel_num[3]
        self.attn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.attn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.attn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.attn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')
        self.attn_norm = LayerNorm3d(self.channel_size, LayerNorm_type='WithBias')

        self.norm1 = nn.LayerNorm(channel_num[0])
        self.norm2 = nn.LayerNorm(channel_num[1])
        self.norm3 = nn.LayerNorm(channel_num[2])
        self.norm4 = nn.LayerNorm(channel_num[3])

        if self.mode == 'kernel':
            self.q1_Conv = nn.Conv2d(channel_num[0], channel_num[0], 1, 1)
            self.k1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)
            self.v1_Conv = nn.Conv2d(self.channel_size, channel_num[0], 1, 1)

            self.q2_Conv = nn.Conv2d(channel_num[1], channel_num[1], 1, 1)
            self.k2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)
            self.v2_Conv = nn.Conv2d(self.channel_size, channel_num[1], 1, 1)

            self.q3_Conv = nn.Conv2d(channel_num[2], channel_num[2], 1, 1)
            self.k3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)
            self.v3_Conv = nn.Conv2d(self.channel_size, channel_num[2], 1, 1)

            self.q4_Conv = nn.Conv2d(channel_num[3], channel_num[3], 1, 1)
            self.k4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)
            self.v4_Conv = nn.Conv2d(self.channel_size, channel_num[3], 1, 1)

            self.out_Conv1 = nn.Conv2d(channel_num[0], channel_num[0], kernel_size=1, bias=False)
            self.out_Conv2 = nn.Conv2d(channel_num[1], channel_num[1], kernel_size=1, bias=False)
            self.out_Conv3 = nn.Conv2d(channel_num[2], channel_num[2], kernel_size=1, bias=False)
            self.out_Conv4 = nn.Conv2d(channel_num[3], channel_num[3], kernel_size=1, bias=False)

            self.softmax = nn.Softmax(dim=1)

        self.ffn_norm1 = LayerNorm3d(channel_num[0], LayerNorm_type='WithBias')
        self.ffn_norm2 = LayerNorm3d(channel_num[1], LayerNorm_type='WithBias')
        self.ffn_norm3 = LayerNorm3d(channel_num[2], LayerNorm_type='WithBias')
        self.ffn_norm4 = LayerNorm3d(channel_num[3], LayerNorm_type='WithBias')

        mlp_ratio = 4
        drop = 0.
        extra_act = True
        self.ffn1 = IRB(in_features=channel_num[0], hidden_features=int(channel_num[0] * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3,
                       extra_act=extra_act)
        self.ffn2 = IRB(in_features=channel_num[1], hidden_features=int(channel_num[1] * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3,
                       extra_act=extra_act)
        self.ffn3 = IRB(in_features=channel_num[2], hidden_features=int(channel_num[2] * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3,
                       extra_act=extra_act)
        self.ffn4 = IRB(in_features=channel_num[3], hidden_features=int(channel_num[3] * mlp_ratio), act_layer=nn.GELU, drop=drop, ksize=3,
                       extra_act=extra_act)

        # self.ffn1 = FeedForward(channel_num[0], ffn_expansion_factor=2.66, bias=False)
        # self.ffn2 = FeedForward(channel_num[1], ffn_expansion_factor=2.66, bias=False)
        # self.ffn3 = FeedForward(channel_num[2], ffn_expansion_factor=2.66, bias=False)
        # self.ffn4 = FeedForward(channel_num[3], ffn_expansion_factor=2.66, bias=False)

        q_pooled_sizes = [16, 16, 16, 16]
        base_pooled_sizes = [11, 8, 6, 4]
        pooled_sizes = [
            [round(temp * q_pooled_sizes[i] / 16. + 0.001) for temp in base_pooled_sizes] for i in range(4)
        ]
        qkv_bias = True

        self.d_convs1 = nn.ModuleList(
            [nn.Conv2d(channel_num[0], channel_num[0], kernel_size=3, stride=1, padding=1, groups=channel_num[0]) for
             temp
             in pooled_sizes[0]])
        self.d_convs2 = nn.ModuleList(
            [nn.Conv2d(channel_num[1], channel_num[1], kernel_size=3, stride=1, padding=1, groups=channel_num[1]) for
             temp
             in pooled_sizes[1]])
        self.d_convs3 = nn.ModuleList(
            [nn.Conv2d(channel_num[2], channel_num[2], kernel_size=3, stride=1, padding=1, groups=channel_num[2]) for
             temp
             in pooled_sizes[2]])
        self.d_convs4 = nn.ModuleList(
            [nn.Conv2d(channel_num[3], channel_num[3], kernel_size=3, stride=1, padding=1, groups=channel_num[3]) for
             temp
             in pooled_sizes[3]])

        self.kv1 = nn.Sequential(nn.Linear(channel_num[0], channel_num[0] * 2, bias=qkv_bias))
        self.kv2 = nn.Sequential(nn.Linear(channel_num[1], channel_num[1] * 2, bias=qkv_bias))
        self.kv3 = nn.Sequential(nn.Linear(channel_num[2], channel_num[2] * 2, bias=qkv_bias))
        self.kv4 = nn.Sequential(nn.Linear(channel_num[3], channel_num[3] * 2, bias=qkv_bias))

        self.scale1 = channel_num[0] // num_heads ** -0.5
        self.scale2 = channel_num[1] // num_heads ** -0.5
        self.scale3 = channel_num[2] // num_heads ** -0.5
        self.scale4 = channel_num[3] // num_heads ** -0.5

        self.eps = 0.001

        self.cpe1 = nn.Conv2d(channel_num[0], channel_num[0], 3, 1, 1, groups=channel_num[0])
        self.cpe2 = nn.Conv2d(channel_num[1], channel_num[1], 3, 1, 1, groups=channel_num[1])
        self.cpe3 = nn.Conv2d(channel_num[2], channel_num[2], 3, 1, 1, groups=channel_num[2])
        self.cpe4 = nn.Conv2d(channel_num[3], channel_num[3], 3, 1, 1, groups=channel_num[3])
        self.cpeall = nn.Conv2d(channel_num[0] + channel_num[1]+ channel_num[2] + channel_num[3],
                                channel_num[0] + channel_num[1]+ channel_num[2] + channel_num[3], 3, 1, 1,
                                groups=channel_num[0] + channel_num[1]+ channel_num[2] + channel_num[3])

        self.drop_path1 = DropPath(0.1)
        self.drop_path2 = DropPath(0.1)
        self.drop_path3 = DropPath(0.1)
        self.drop_path4 = DropPath(0.1)

    def forward(self, emb1, emb2, emb3, emb4):
        b, c, h, w = emb1.shape
        emb_all = torch.cat([emb1, emb2, emb3, emb4], dim=1)

        emb1 = self.cpe1(emb1) + emb1
        emb2 = self.cpe2(emb2) + emb2
        emb3 = self.cpe3(emb3) + emb3
        emb4 = self.cpe4(emb4) + emb4
        emb_all = self.cpeall(emb_all) + emb_all

        emb1_norm = self.attn_norm1(emb1)
        emb2_norm = self.attn_norm2(emb2)
        emb3_norm = self.attn_norm3(emb3)
        emb4_norm = self.attn_norm4(emb4)
        emb_all = self.attn_norm(emb_all)

        if self.mode == 'kernel':
            emb1_gq = self.q1_Conv(emb1_norm)
            emb_all_gk1 = self.k1_Conv(emb_all)
            emb_all_gv1 = self.v1_Conv(emb_all)
            emb1_gq = rearrange(emb1_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk1 = rearrange(emb_all_gk1, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv1 = rearrange(emb_all_gv1, 'b c h w -> b c (h w)', h=h, w=w)
            emb1_gq, emb_all_gk1 = (F.normalize(emb1_gq, dim=-1), F.normalize(emb_all_gk1, dim=-1))

            emb2_gq = self.q2_Conv(emb2_norm)
            emb_all_gk2 = self.k2_Conv(emb_all)
            emb_all_gv2 = self.v2_Conv(emb_all)
            emb2_gq = rearrange(emb2_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk2 = rearrange(emb_all_gk2, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv2 = rearrange(emb_all_gv2, 'b c h w -> b c (h w)', h=h, w=w)
            emb2_gq, emb_all_gk2 = (F.normalize(emb2_gq, dim=-1), F.normalize(emb_all_gk2, dim=-1))

            emb3_gq = self.q3_Conv(emb3_norm)
            emb_all_gk3 = self.k3_Conv(emb_all)
            emb_all_gv3 = self.v3_Conv(emb_all)
            emb3_gq = rearrange(emb3_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk3 = rearrange(emb_all_gk3, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv3 = rearrange(emb_all_gv3, 'b c h w -> b c (h w)', h=h, w=w)
            emb3_gq, emb_all_gk3 = (F.normalize(emb3_gq, dim=-1), F.normalize(emb_all_gk3, dim=-1))

            emb4_gq = self.q4_Conv(emb4_norm)
            emb_all_gk4 = self.k4_Conv(emb_all)
            emb_all_gv4 = self.v4_Conv(emb_all)
            emb4_gq = rearrange(emb4_gq, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gk4 = rearrange(emb_all_gk4, 'b c h w -> b c (h w)', h=h, w=w)
            emb_all_gv4 = rearrange(emb_all_gv4, 'b c h w -> b c (h w)', h=h, w=w)
            emb4_gq, emb_all_gk4 = (F.normalize(emb4_gq, dim=-1), F.normalize(emb_all_gk4, dim=-1))

            # Conduct Pyramid Pooling on K, V
            pools1 = []
            pools2 = []
            pools3 = []
            pools4 = []
            c1 = emb_all_gk1.shape[1]
            c2 = emb_all_gk2.shape[1]
            c3 = emb_all_gk3.shape[1]
            c4 = emb_all_gk4.shape[1]
            emb1_gq_res = emb1_gq.transpose(1, 2).reshape(b, -1, self.num_heads,
                                                          emb1_gq.shape[1] // self.num_heads).permute(0, 2, 1,
                                                                                                      3).contiguous()
            emb2_gq_res = emb2_gq.transpose(1, 2).reshape(b, -1, self.num_heads,
                                                          emb2_gq.shape[1] // self.num_heads).permute(0, 2, 1,
                                                                                                      3).contiguous()
            emb3_gq_res = emb3_gq.transpose(1, 2).reshape(b, -1, self.num_heads,
                                                          emb3_gq.shape[1] // self.num_heads).permute(0, 2, 1,
                                                                                                      3).contiguous()
            emb4_gq_res = emb4_gq.transpose(1, 2).reshape(b, -1, self.num_heads,
                                                          emb4_gq.shape[1] // self.num_heads).permute(0, 2, 1,
                                                                                                      3).contiguous()
            x1_ = emb_all_gk1.reshape(b, c1, h, w)
            x2_ = emb_all_gk2.reshape(b, c2, h, w)
            x3_ = emb_all_gk3.reshape(b, c3, h, w)
            x4_ = emb_all_gk4.reshape(b, c4, h, w)
            for (pooled_size, l1, l2, l3, l4) in zip(self.base_pooled_sizes, self.d_convs1, self.d_convs2,
                                                     self.d_convs3, self.d_convs4):
                pooled_size = (pooled_size, round(w * pooled_size / h + self.eps)) if w >= h else (
                    round(h * pooled_size / w + self.eps), pooled_size)

                pool1 = F.adaptive_avg_pool2d(x1_, pooled_size)
                pool1 = pool1 + l1(pool1)
                pools1.append(pool1.view(b, c1, -1))

                pool2 = F.adaptive_avg_pool2d(x2_, pooled_size)
                pool2 = pool2 + l2(pool2)
                pools2.append(pool2.view(b, c2, -1))

                pool3 = F.adaptive_avg_pool2d(x3_, pooled_size)
                pool3 = pool3 + l3(pool3)
                pools3.append(pool3.view(b, c3, -1))

                pool4 = F.adaptive_avg_pool2d(x4_, pooled_size)
                pool4 = pool4 + l4(pool4)
                pools4.append(pool4.view(b, c4, -1))

            pools1 = torch.cat(pools1, dim=2)
            pools1 = self.norm1(pools1.permute(0, 2, 1))
            kv1 = self.kv1(pools1).reshape(b, -1, 2, self.num_heads, c1 // self.num_heads).permute(2, 0, 3, 1, 4)
            k1, v1 = kv1[0], kv1[1]

            pools2 = torch.cat(pools2, dim=2)
            pools2 = self.norm2(pools2.permute(0, 2, 1))
            kv2 = self.kv2(pools2).reshape(b, -1, 2, self.num_heads, c2 // self.num_heads).permute(2, 0, 3, 1, 4)
            k2, v2 = kv2[0], kv2[1]

            pools3 = torch.cat(pools3, dim=2)
            pools3 = self.norm3(pools3.permute(0, 2, 1))
            kv3 = self.kv3(pools3).reshape(b, -1, 2, self.num_heads, c3 // self.num_heads).permute(2, 0, 3, 1, 4)
            k3, v3 = kv3[0], kv3[1]

            pools4 = torch.cat(pools4, dim=2)
            pools4 = self.norm4(pools4.permute(0, 2, 1))
            kv4 = self.kv4(pools4).reshape(b, -1, 2, self.num_heads, c4 // self.num_heads).permute(2, 0, 3, 1, 4)
            k4, v4 = kv4[0], kv4[1]

            # self-attention
            attn1 = (emb1_gq_res @ k1.transpose(-2, -1)) * self.scale1
            attn1 = attn1.softmax(dim=-1)
            out1 = (attn1 @ v1)  # B N C
            out1 = out1.transpose(-1, -2).flatten(1, 2)

            attn2 = (emb2_gq_res @ k2.transpose(-2, -1)) * self.scale2
            attn2 = attn2.softmax(dim=-1)
            out2 = (attn2 @ v2)
            out2 = out2.transpose(-1, -2).flatten(1, 2)

            attn3 = (emb3_gq_res @ k3.transpose(-2, -1)) * self.scale3
            attn3 = attn3.softmax(dim=-1)
            out3 = (attn3 @ v3)
            out3 = out3.transpose(-1, -2).flatten(1, 2)

            attn4 = (emb4_gq_res @ k4.transpose(-2, -1)) * self.scale4
            attn4 = attn4.softmax(dim=-1)
            out4 = (attn4 @ v4)
            out4 = out4.transpose(-1, -2).flatten(1, 2)
            ##################################################################

            # sim1 = self.kernel_similarity(emb1_gq, emb_all_gk1)
            # sim2 = self.kernel_similarity(emb2_gq, emb_all_gk2)
            # sim3 = self.kernel_similarity(emb3_gq, emb_all_gk3)
            # sim4 = self.kernel_similarity(emb4_gq, emb_all_gk4)
            #
            # sim1, sim2, sim3, sim4 = self.softmax(sim1), self.softmax(sim2), self.softmax(sim3), self.softmax(sim4)
            #
            # out1 = sim1 * emb_all_gv1
            # out2 = sim2 * emb_all_gv2
            # out3 = sim3 * emb_all_gv3
            # out4 = sim4 * emb_all_gv4

            out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h, w=w)
            out2 = rearrange(out2, 'b c (h w) -> b c h w', h=h, w=w)
            out3 = rearrange(out3, 'b c (h w) -> b c h w', h=h, w=w)
            out4 = rearrange(out4, 'b c (h w) -> b c h w', h=h, w=w)

            att_out1 = self.drop_path1(out1)
            att_out2 = self.drop_path2(out2)
            att_out3 = self.drop_path3(out3)
            att_out4 = self.drop_path4(out4)

            # att_out1 = self.out_Conv1(out1)
            # att_out2 = self.out_Conv2(out2)
            # att_out3 = self.out_Conv3(out3)
            # att_out4 = self.out_Conv4(out4)

            att_out1 = att_out1 + emb1
            att_out2 = att_out2 + emb2
            att_out3 = att_out3 + emb3
            att_out4 = att_out4 + emb4

            res1 = att_out1
            res2 = att_out2
            res3 = att_out3
            res4 = att_out4
            x1 = self.ffn_norm1(res1)
            x2 = self.ffn_norm2(res2)
            x3 = self.ffn_norm3(res3)
            x4 = self.ffn_norm4(res4)
            x1 = self.drop_path1(self.ffn1(x1))
            x2 = self.drop_path2(self.ffn2(x2))
            x3 = self.drop_path3(self.ffn3(x3))
            x4 = self.drop_path4(self.ffn4(x4))
            x1 = x1 + res1
            x2 = x2 + res2
            x3 = x3 + res3
            x4 = x4 + res4

            return x1, x2, x3, x4


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.layer_norm1 = LayerNorm3d(hidden_features * 2, LayerNorm_type='WithBias')
        self.act = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.layer_norm1(x)
        x = self.act(x)
        x = self.project_out(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out


class CAHighPassDWT(nn.Module):
    def __init__(self, in_planes, ratio=16, Mratio=None, atten_factor=0.1):
        super().__init__()
        self.Mratio = Mratio
        self.atten_factor = atten_factor
        # --- 1. DCT 分支的组件 ---
        # 专门处理高频DCT信号的 MLP
        self.dct_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_planes * 2, in_planes // ratio * 2),
            nn.ReLU(),
            nn.Linear(in_planes // ratio * 2, in_planes)
        )

        # --- 2. DWT 分支的组件 ---
        self.wavelet = DWTForward(J=1, wave='haar', mode='symmetric')

        # 融合 LL 和 3个高频带 (LH, HL, HH)
        # 输入维度是 in_planes * 2 (LL占in_planes, H占in_planes)
        self.dwt_fusion_conv = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes, 1),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )

        # 专门处理DWT信号的 MLP
        self.ll_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_planes, in_planes // ratio),  # 输入是 C
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)  # 输出是 C
        )
        self.high_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_planes, in_planes // ratio),  # 输入是 C
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes)  # 输出是 C
        )

        # --- 3. 通用组件 ---
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 最终融合权重的激活
        # 注意：这里我不加Sigmoid，因为两个分支已经分别Sigmoid了，最后是一个加权过程
        self.channel_fusion = nn.Conv1d(2, 1, kernel_size=1, bias=False)

        self.norm_ll = nn.BatchNorm2d(in_planes)
        self.norm_high = nn.BatchNorm2d(in_planes)
        # self.norm_ll = nn.InstanceNorm2d(in_planes, affine=True)
        # self.norm_high = nn.InstanceNorm2d(in_planes, affine=True)  # 或者不用affine

        self.gamma_dct = nn.Parameter(torch.full((1,), 0.5))
        self.gamma_dwt = nn.Parameter(torch.full((1,), 0.5))

    def _compute_dct_weight(self, h, w, ratio):
        """ 生成高通滤波掩码 """
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = self.atten_factor
        return weight

    def forward(self, x):
        b, c, h, w = x.size()

        # ==========================================
        # Branch 1: DCT (Global High Frequency)
        # ==========================================
        if self.Mratio is not None:
            # 1. 转频域
            dct_x = dct_2d(x, norm="ortho")

            # 2. 高通滤波 (Masking)
            mask = self._compute_dct_weight(h, w, self.Mratio).to(x.device)
            mask = mask.view(1, 1, h, w)
            m_dct_x = dct_x * mask

            # 3. 转回空域 (此时 idct_x 近似为残差图/边缘图)
            idct_x = idct_2d(m_dct_x, norm='ortho')
            # --- 关键改动：高频信号池化策略 ---
            # 策略A: MaxPool (捕捉最强异常点，如高亮小目标)

            # 策略B: StdPool (标准差池化，捕捉通道的能量/纹理丰富度)
            # 解释：高通图均值约为0，计算Std比Avg更能代表该通道是否包含大量高频信息
            # 计算公式：std = sqrt(mean((x - mean)^2))
            dct_std = torch.std(idct_x.reshape(b, c, -1), dim=2).reshape(b, c, 1, 1)
            dct_max = self.global_max_pool(idct_x.abs())
        else:
            idct_x = x
            dct_std = self.global_avg_pool(idct_x)
            dct_max = self.global_max_pool(idct_x)

        # 融合两种池化 (相加)
        dct_desc = torch.cat((dct_max, dct_std), dim=1)
        # dct_desc = dct_max + dct_std #后续改成cat

        # 计算 DCT 注意力向量
        dct_att = self.dct_mlp(dct_desc).view(b, c, 1, 1)

        # ==========================================
        # Branch 2: DWT (Local High Frequency)
        # ==========================================
        # 1. 小波变换
        ll, H = self.wavelet(x)
        # ll shape: [B, C, H/2, W/2]
        # H[0] shape: [B, C, 3, H/2, W/2]

        # 2. 提取高频并取绝对值 (关键！)
        # 我们关注的是“有没有边缘”，而不是边缘是黑白还是白黑
        lh = torch.abs(H[0][:, :, 0, :, :])
        hl = torch.abs(H[0][:, :, 1, :, :])
        hh = torch.abs(H[0][:, :, 2, :, :])

        # 3. 合并高频分量 (这里用Max，表示取三个方向中最强的边缘)
        high_freq_combined = torch.max(torch.stack([lh, hl, hh], dim=0), dim=0)[0]

        # --- 关键修改：先归一化，再 Cat ---
        ll = self.norm_ll(ll)
        high = self.norm_high(high_freq_combined)

        # 4. 与 LL 低频分量拼接
        # dwt_input = torch.cat([ll_norm, high_norm], dim=1)  # [B, 2C, H/2, W/2]

        # 5. 卷积融合特征
        # dwt_feat = self.dwt_fusion_conv(dwt_input)  # [B, C, H/2, W/2]

        # 6. DWT 分支池化 (LL包含背景，所以AvgPool在这里是有效的)
        ll_desc = self.global_avg_pool(ll) + self.global_max_pool(ll)
        high_desc = self.global_avg_pool(high) + self.global_max_pool(high)

        ll_logits = self.ll_mlp(ll_desc).view(b, c, 1, 1)
        high_logits = self.high_mlp(high_desc).view(b, c, 1, 1)
        dwt_att = ll_logits + high_logits

        # ==========================================
        # Final Fusion
        # ==========================================
        # 此时我们有两个注意力向量：
        # dct_att: [B, C, 1, 1] (0~1之间)
        # dwt_att: [B, C, 1, 1] (0~1之间)

        # 简单相乘通常会导致数值过小，这里建议使用“最大化”或“加权和”
        # 方案：直接相加后截断，或者取Max
        # 这里使用加和，表示只要在一个频域被认为是重要的，它就是重要的
        logits = self.gamma_dct * dct_att + self.gamma_dwt * dwt_att
        logits = torch.tanh(logits)
        final_att = torch.sigmoid(logits)

        # 如果你想让网络自己学习怎么融合这两个域，可以用下面这行代替上面那行：
        # stacked = torch.stack([dct_att, dwt_att], dim=1).squeeze(-1).squeeze(-1) # [B, 2, C]
        # final_att = self.channel_fusion(stacked).view(b, c, 1, 1).sigmoid()

        # ==========================================
        # !!! 必须乘回原图 !!!
        # ==========================================
        return final_att


class DyHighPassAttn(BaseModule):
    def __init__(self,
                 in_channels=16,
                 isdct=True,
                 highpass_kernel=3,  # 高通滤波器 kernel 大小
                 up_group=1,  # carafe 分组数
                 compressed_channels=64,  # 通道压缩维度
                 encoder_kernel=3,  # 掩码生成器卷积核大小
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)
        self.isdct = isdct  # true when in p1&p2; false when in p3&p4
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.compressed_channels = compressed_channels

        # 通道压缩器：将输入通道压缩到指定维度
        self.hr_channel_compressor = nn.Conv2d(in_channels, compressed_channels, 1)

        self.huifuConv = nn.Conv2d(self.compressed_channels, in_channels, kernel_size=1)

        # 动态掩码生成器（AHPF 生成器）：生成 carafe 需要的掩码
        self.content_encoder2 = nn.Conv2d(
            compressed_channels,
            highpass_kernel ** 2 * up_group * 1 * 1,  # 1x1 上采样（保持尺寸）
            encoder_kernel,
            padding=int((encoder_kernel - 1) / 2),  # 保持尺寸
            groups=1
        )

        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
                *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
            )

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化卷积层权重
        if hasattr(self, 'content_encoder2'):
            nn.init.xavier_normal_(self.content_encoder2.weight, gain=1.0)
            if self.content_encoder2.bias is not None:
                nn.init.constant_(self.content_encoder2.bias, 0)
        if hasattr(self, 'hr_channel_compressor'):
            nn.init.xavier_normal_(self.hr_channel_compressor.weight, gain=1.0)
            if self.hr_channel_compressor.bias is not None:
                nn.init.constant_(self.hr_channel_compressor.bias, 0)
        self.sa = SpatialAttention()
    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))  # group
        # mask = mask.view(n, mask_channel, -1, h, w)
        # mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        # mask = mask.view(n, mask_c, h, w).contiguous()

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        # mask = F.pad(mask, pad=[padding] * 4, mode=self.padding_mode) # kernel + 2 * padding
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            # 非 DCT 模式保持不变
            return x * torch.sigmoid(self.spatial1x1(x))

        # 高通滤波替换为 carafe 实现
        # 1. 压缩通道维度
        compressed_hr_feat = self.hr_channel_compressor(x)
        # 2. 生成高通掩码（AHPF）
        mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
        # 3. 掩码归一化
        mask_hr_init = self.kernel_normalizer(
            mask_hr_hr_feat,
            self.highpass_kernel
        )
        # 4. 计算高通特征：原始特征 - 低通特征（carafe 生成低通）
        highpass_feat = compressed_hr_feat - carafe(
            compressed_hr_feat,
            mask_hr_init,
            self.highpass_kernel,
            self.up_group,
            1  # 保持尺寸不变（scale=1）
        )
        return self.sa(highpass_feat)

class Res_CBAM_block_wConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mratio=(0.025, 0.025), atten_factor=0.1,isdct=True):
        super().__init__()
        self.conv1 = wConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, den=den)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = wConv2d(out_channels, out_channels, kernel_size=5, padding=2, den=[0.3, 0.5])
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        # self.ca = CAHighPassDWT(out_channels, Mratio=mratio, atten_factor=atten_factor)
        self.ca = ChannelAttention(out_channels)
        # self.sa = DyHighPassAttn(out_channels,isdct=isdct)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * self.ca(out)
        out = out * self.sa(out)
        out += residual
        out = self.relu(out)
        return out

class Res_CBAM_block_hpf_cahpdwt(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mratio=(0.025, 0.025), atten_factor=0.1,isdct=True):
        super().__init__()
        self.conv1 = wConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, den=den)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = wConv2d(out_channels, out_channels, kernel_size=3, padding=1, den=den)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = CAHighPassDWT(out_channels, Mratio=mratio, atten_factor=atten_factor)
        # self.ca = ChannelAttention(out_channels)
        self.sa = DyHighPassAttn(out_channels,isdct=isdct)
        # self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * (1 + self.ca(out))
        out = out * (1 + self.sa(out))
        out += residual
        out = self.relu(out)
        return out

class Res_CBAM_block_hpf(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mratio=(0.025, 0.025), atten_factor=0.1,isdct=True):
        super().__init__()
        self.conv1 = wConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, den=den)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = wConv2d(out_channels, out_channels, kernel_size=5, padding=2, den=[0.3, 0.5])
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        # self.ca = CAHighPassDWT(out_channels, Mratio=mratio, atten_factor=atten_factor)
        self.ca = ChannelAttention(out_channels)
        # self.sa = DyHighPassAttn(out_channels,isdct=isdct)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out * (1 + self.ca(out))
        out = out * self.sa(out)
        out += residual
        out = self.relu(out)
        return out

class CrsSpaAttnWithHighFreqBypass(nn.Module):
        """带高频旁路的跳跃连接"""

        def __init__(self, channel_num=[24, 48, 96, 192], patchSize=[32, 16, 8, 4], layer_num=2, size=256):
            super().__init__()
            # 原始 CrsSpaAttn
            self.original_crsspa = CrsSpaAttn(channel_num, patchSize, layer_num, size)

            # 高频旁路 - 保护小目标高频信息
            self.hf_bypass = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1, groups=c),  # 深度可分离卷积
                    nn.Conv2d(c, c, 1),
                    nn.BatchNorm2d(c),
                    nn.Sigmoid()
                ) for c in channel_num
            ])

            # 自适应融合权重
            self.fusion_weights = nn.Parameter(torch.tensor([0.7, 0.3]))  # [原始权重, 高频权重]

        def forward(self, en1, en2, en3, en4):
            # 原始处理
            out1, out2, out3, out4 = self.original_crsspa(en1, en2, en3, en4)

            # 高频旁路
            hf1 = self.hf_bypass[0](en1)
            hf2 = self.hf_bypass[1](en2)
            hf3 = self.hf_bypass[2](en3)
            hf4 = self.hf_bypass[3](en4)

            # 加权融合
            w = torch.softmax(self.fusion_weights, dim=0)
            out1 = w[0] * out1 + w[1] * (en1 * hf1)
            out2 = w[0] * out2 + w[1] * (en2 * hf2)
            out3 = w[0] * out3 + w[1] * (en3 * hf3)
            out4 = w[0] * out4 + w[1] * (en4 * hf4)

            return out1, out2, out3, out4

class SFDANet(nn.Module):
    def __init__(self, input_channels, block=ResNet, size=256):
        super().__init__()
        param_channels = [24, 48, 96, 192]  # down3
        param_blocks = [2, 2, 2, 2]
        size = size
        self.pool1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(param_channels[0], param_channels[0] * 2, kernel_size=1, stride=1))
        self.pool2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(param_channels[1], param_channels[1] * 2, kernel_size=1, stride=1))
        self.pool3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(param_channels[2], param_channels[2] * 2, kernel_size=1, stride=1))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.stage_configs = [
            {'mratio': (0.125, 0.125), 'atten': 0.2},  # Stage 0 - 更弱
            {'mratio': (0.125, 0.125), 'atten': 0.4},  # Stage 1 - 更弱
            {'mratio': (0.125, 0.125), 'atten': 0.6},  # Stage 2 - 略微增强
            {'mratio': (0.125, 0.125), 'atten': 0.8},  # Stage 3 - 最强
        ]

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], Res_CBAM_block_hpf_cahpdwt, param_blocks[0], config=self.stage_configs[0])
        self.encoder_1 = self._make_layer(param_channels[0] * 2, param_channels[1], Res_CBAM_block_hpf_cahpdwt, param_blocks[1], config=self.stage_configs[1])
        self.encoder_2 = self._make_layer(param_channels[1] * 2, param_channels[2], Res_CBAM_block_hpf_cahpdwt,
                                          param_blocks[2], config=self.stage_configs[2])
        self.encoder_3 = self._make_layer(param_channels[2] * 2, param_channels[3], Res_CBAM_block_hpf_cahpdwt,
                                          param_blocks[3], config=self.stage_configs[3])

        # self.crsspa = CrsSpaAttn(
        #     channel_num=[param_channels[0], param_channels[1], param_channels[2], param_channels[3]],
        #     patchSize=[32, 16, 8, 4], layer_num=2, size=size)

        self.HighRidePath = CrsSpaAttnWithHighFreqBypass(
            channel_num=param_channels,
            patchSize=[32, 16, 8, 4],
            layer_num=2,
            size=size
        )

        self.decoder_2 = self._make_layer(param_channels[3] + param_channels[2], param_channels[2], block,
                                          param_blocks[2])
        self.decoder_1 = self._make_layer(param_channels[2] + param_channels[1], param_channels[1], block,
                                          param_blocks[1])
        self.decoder_0 = self._make_layer(param_channels[1] + param_channels[0], param_channels[0], block,
                                          param_blocks[0])

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.final = nn.Conv2d(3, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1, config=None):
        layer = []
        if config is None:
            layer.append(block(in_channels, out_channels))
            for _ in range(block_num - 1):
                layer.append(block(out_channels, out_channels))
            return nn.Sequential(*layer)
        else:
            layer.append(block(in_channels, out_channels, mratio=config['mratio'], atten_factor=config['atten']))
            for _ in range(block_num - 1):
                layer.append(block(out_channels, out_channels, mratio=config['mratio'], atten_factor=config['atten']))
            return nn.Sequential(*layer)

    def forward(self, x, tag=True):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool1(x_e0))
        x_e2 = self.encoder_2(self.pool2(x_e1))
        x_m = self.encoder_3(self.pool3(x_e2))

        # x_e0, x_e1, x_e2, x_m = self.crsspa(x_e0, x_e1, x_e2, x_m)

        x_e0, x_e1, x_e2, x_m = self.HighRidePath(x_e0, x_e1, x_e2, x_m)


        x_d2 = self.decoder_2(torch.cat((x_e2, self.up(x_m)), dim=1))
        x_d1 = self.decoder_1(torch.cat((x_e1, self.up(x_d2)), dim=1))
        x_d0 = self.decoder_0(torch.cat((x_e0, self.up(x_d1)), dim=1))

        if tag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2)], dim=1))
            mask1 = F.interpolate(mask1, scale_factor=2, mode='bilinear', align_corners=False)
            mask2 = F.interpolate(mask2, scale_factor=4, mode='bilinear', align_corners=False)
            return [output, mask1, mask2, mask0]

        else:
            output = self.output_0(x_d0)
            return [], output


if __name__ == "__main__":
    # 设置参数
    crop_size = 256  # 假设输入是 256x256

    # 创建模型
    model = SFDANet(1)

    # 将模型设置为评估模式（虽然对于thop计算不是强制的，但在逻辑上是严谨的）
    model.eval()

    print("=" * 50)
    print("正在计算模型参数量和 FLOPs...")

    # --------------------------------------------------------
    # 使用 thop 计算 Params 和 FLOPs
    # --------------------------------------------------------
    # 1. 创建一个用于测试计算量的 dummy input (通常使用 batch_size=1)
    input_test = torch.randn(4, 1, crop_size, crop_size)

    # 2. 调用 profile
    # 注意：inputs 是一个元组，对应 forward 函数的参数。
    # 如果你的 forward(x, deep_supervision) 需要第二个参数，这里也需要传入。
    # 这里我假设计算量时也传入 True，或者你可以根据推理阶段的需求传入 False
    flops, params = profile(model, inputs=(input_test, True), verbose=False)

    # 3. 使用 clever_format 进行格式化 (例如将 1024000 转为 1.024M)
    flops_str, params_str = clever_format([flops, params], "%.3f")

    print(f"Params: {params_str}")
    print(f"FLOPs:  {flops_str}")
    print("=" * 50)

    # --------------------------------------------------------
    # 原有的输出形状测试代码
    # --------------------------------------------------------
    test_shapes = [
        (4, 1, crop_size, crop_size),  # batch=4
    ]

    for shape in test_shapes:
        x = torch.randn(shape)
        with torch.no_grad():
            output = model(x, True)

        print(f"\n当前测试 Batch Shape: {shape}")

        if isinstance(output, list):
            # 情况2: 只返回 masks 列表
            print(f"模型返回: masks列表")
            print(f"列表长度: {len(output)}")
            for i, mask in enumerate(output):
                print(f"  mask{i}: {mask.shape}")
        elif isinstance(output, torch.Tensor):
            # 情况3: 只返回单个张量
            print(f"模型返回: 单个张量")
            print(f"Output: {output.shape}")
        else:
            # 其他情况
            print(f"未知输出类型: {type(output)}")
            print(f"输出内容: {output}")
