# import ipdb
import torch

from einops import rearrange
from functools import partial
from torch import nn, einsum



class Agg_0(nn.Module):
    def __init__(self, seg_dim):
        super().__init__()
        self.conv = SeparableConv2d(seg_dim * 3, seg_dim, 3, 1, 1)
        self.norm = nn.LayerNorm(seg_dim)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = self.act(self.norm(x.reshape(b, c, -1).permute(0, 2, 1)))

        return x


class Aggregator(nn.Module):
    def __init__(self, dim, seg=4):
        super().__init__()
        self.dim = dim
        self.seg = seg

        seg_dim = self.dim // self.seg


        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(seg_dim)
        self.act1 = nn.Hardswish()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.BatchNorm2d(seg_dim)
        self.act2 = nn.Hardswish()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 7, 1, 3)
        self.norm3 = nn.BatchNorm2d(seg_dim)
        self.act3 = nn.Hardswish()

        self.agg0 = Agg_0(seg_dim)


    def forward(self, x, size, num_head):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        seg_dim = self.dim // self.seg

        x = x.split([seg_dim]*self.seg, dim=1)

        x_local = x[3].reshape(3, B//3, seg_dim, H, W).permute(1,0,2,3,4).reshape(B//3, 3*seg_dim, H, W)
        x_local = self.agg0(x_local)

        x1 = self.act1(self.norm1(self.agg1(x[0])))
        x2 = self.act2(self.norm2(self.agg2(x[1])))
        x3 = self.act3(self.norm3(self.agg3(x[2])))

        x = torch.cat([x1, x2, x3], dim=1)

        C = C // 4 * 3
        x = x.reshape(3, B//3, num_head, C//num_head, H*W).permute(0, 1, 2, 4, 3)

        return x, x_local


class ConvRelPosEnc(nn.Module):
    """ Convolutional relative position encoding. """

    def __init__(self, Ch, h, window):
        super().__init__()

        if isinstance(window, int):
            window = {window: h} 
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:
            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  
            padding_size = (cur_window + (cur_window - 1) * (dilation - 1)) // 2
            cur_conv = nn.Conv2d(cur_head_split * Ch, cur_head_split * Ch,
                                 kernel_size=(cur_window, cur_window),
                                 padding=(padding_size, padding_size),
                                 dilation=(dilation, dilation),
                                 groups=cur_head_split * Ch,
                                 )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)

        self.channel_splits = [x * Ch for x in self.head_splits]

    def forward(self, q, v, size):

        B, h, N, Ch = q.shape
        H, W = size
        assert N == H * W

        q_img = q  
        v_img = v  

        v_img = rearrange(v_img, 'B h (H W) Ch -> B (h Ch) H W', H=H, W=W)  # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels.
        conv_v_img_list = [conv(x) for conv, x in zip(self.conv_list, v_img_list)]
        conv_v_img = torch.cat(conv_v_img_list, dim=1)
        conv_v_img = rearrange(conv_v_img, 'B (h Ch) H W -> B h (H W) Ch', h=h)  # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].

        EV_hat_img = q_img * conv_v_img

        return EV_hat_img


class EfficientAtt(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregator = Aggregator(dim=dim, seg=4)

        trans_dim = dim // 4 * 3
        self.crpe = ConvRelPosEnc(Ch=trans_dim // num_heads, h=num_heads, window={3: 2, 5: 3, 7: 3})

    def forward(self, x,seg, size):
        B, C, H, W = x.size()
        x = x.reshape(B,H*W,C)

        seg = seg.reshape(B, H * W, C)
        B, N, C = x.shape

        kv = self.kv(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).reshape(2*B, N, C)
        q = self.q(seg).reshape(B, N, 1, C).permute(2, 0, 1, 3).reshape(B, N, C)

        qkv = torch.cat([q,kv],dim=0)
        qkv, x_agg0 = self.aggregator(qkv, size, self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_softmax = k.softmax(dim=2)
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)
        eff_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)
        crpe = self.crpe(q, v, size=size)

        x = self.scale * eff_att + crpe
        x = x.transpose(1, 2).reshape(B, N, C//4*3)
        x = torch.cat([x, x_agg0], dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x

