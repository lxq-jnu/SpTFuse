import torch
import torch.nn.functional as F
import torch.nn as nn
from .groupmixattention import EfficientAtt

class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(2, 0, 1).contiguous()
        x1 = torch.cat([v_i[:, :, None] for i, v_i in enumerate(x) if i % 2 == 0], dim=-1)
        x2 = torch.cat([v_i[:, :, None] for i, v_i in enumerate(x) if i % 2 == 1], dim=-1)

        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2


class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim_in,dim_out, nhead=8, dim_feedforward=2048, dropout=0.1,layer_index = 1):
        super(TransformerEncoderBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.act_layer = nn.GELU
        self.SCFN_ffn = SGFN(in_features=self.dim_out, hidden_features=self.dim_out, out_features=self.dim_out, act_layer=self.act_layer)

        self.norm1 = nn.LayerNorm(self.dim_out)
        self.norm2 = nn.LayerNorm(self.dim_out)
        self.dropout = nn.Dropout(dropout)
        self.conv1x1_in = nn.Conv2d(self.dim_in, self.dim_out, kernel_size=1)

        self.conv1x1_out = nn.Conv2d(self.dim_out,self.dim_in, kernel_size=1)
        self.self_attn_group = EfficientAtt(num_heads=nhead, attn_drop=0., proj_drop=0.)


    def forward(self, x,seg,C):

        b,c,h,w=x.size()
        flag = 0
        if c != C:
            x = self.conv1x1_in(x)
            flag = 1

        attn_output = self.self_attn_group(x,seg,(h,w))

        x_in = x.view(b,-1,C)
        x = x_in + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.SCFN_ffn(x,h,w)
        ffn_output1 = 0.5 * ffn_output + 0.5 * ffn_output.mean(dim=1,keepdim=True)
        x = x + self.dropout(ffn_output1)
        x = self.norm2(x)

        x = x.reshape(b, C, h, w)
        if flag:
            x = self.conv1x1_out(x)
        return x

class DualTransformerEncoderBlock(nn.Module):
    def __init__(self, dim_in,dim_out, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(DualTransformerEncoderBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.act_layer = nn.GELU
        self.SCFN_ffn1 = SGFN(in_features=self.dim_out, hidden_features=self.dim_out, out_features=self.dim_out, act_layer=self.act_layer)
        self.SCFN_ffn2 = SGFN(in_features=self.dim_out, hidden_features=self.dim_out, out_features=self.dim_out, act_layer=self.act_layer)

        self.norm1 = nn.LayerNorm(self.dim_out)
        self.norm2 = nn.LayerNorm(self.dim_out)
        self.norm3 = nn.LayerNorm(self.dim_out)
        self.norm4 = nn.LayerNorm(self.dim_out)
        self.dropout = nn.Dropout(dropout)
        self.conv1x1_in = nn.Conv2d(self.dim_in, self.dim_out, kernel_size=1)

        self.conv1x1_out = nn.Conv2d(self.dim_out,self.dim_in, kernel_size=1)
        self.con3x3 = nn.Sequential(nn.Conv2d(in_channels=self.dim_out*2, out_channels=self.dim_out, kernel_size=3,padding=1),
                                    nn.ReLU()
                                    )

        self.self_attn_group1 = EfficientAtt( num_heads=nhead, attn_drop=0., proj_drop=0.,)
        self.self_attn_group2 = EfficientAtt( num_heads=nhead, attn_drop=0., proj_drop=0.,)


    def forward(self, x,seg,C):

        b,c,h,w=x.size()
        flag = 0
        if c != C:
            x = self.conv1x1_in(x)  
            flag = 1

        attn_output = self.self_attn_group1(x,seg,(h,w))

        x_in = x.view(b,-1,C)
        x = x_in + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output1 = self.SCFN_ffn1(x,h,w)
        ffn_output1 = 0.5 * ffn_output1 + 0.5 * ffn_output1.mean(dim=1,keepdim=True)
        x = x + self.dropout(ffn_output1)
        x = self.norm2(x)
        x = x.reshape(b, C, h, w)
        x_one_out1 = x


        attn_output = self.self_attn_group2(x,seg,(h,w))

        x_in = x.view(b,-1,C)
        x = x_in + self.dropout(attn_output)
        x = self.norm3(x)


        ffn_output2 = self.SCFN_ffn2(x,h,w)
        ffn_output2 = 0.5 * ffn_output2 + 0.5 * ffn_output2.mean(dim=1,keepdim=True)
        x = x + self.dropout(ffn_output2)
        x = self.norm4(x)
        x = x.reshape(b, C, h, w)
        x_one_out2 = x

        x_gat = torch.cat([x_one_out1, x_one_out2], dim=1)
        x = self.con3x3(x_gat)

        x = x.reshape(b, C, h, w)
        if flag:
            x = self.conv1x1_out(x)
        return x



