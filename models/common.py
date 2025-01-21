import torch
from torch import nn
import torch.nn.functional as F

class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient



def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out

class MIM(nn.Module):
    def __init__(self, feature_channel,out_channel=32):
        super(MIM, self).__init__()
        self.scale = nn.Sequential(nn.Conv2d(feature_channel, out_channel, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(out_channel, feature_channel, 1))
        self.shift = nn.Sequential(nn.Conv2d(feature_channel, out_channel, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(out_channel, feature_channel, 1))


    def forward(self, feature_in, condition):
        scale = self.scale(condition)
        shift = self.shift(condition)
        feature_out = torch.mul(feature_in, (scale+1)) + shift
        return feature_out
class SEM(nn.Module):
    def __init__(self, feature_channel,out_channel=32):
        super(SEM, self).__init__()
        self.scale = nn.Sequential(nn.Conv2d(feature_channel, feature_channel * 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(feature_channel * 2, out_channel, 1))
        self.shift = nn.Sequential(nn.Conv2d(feature_channel, feature_channel * 2, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(feature_channel * 2, out_channel, 1))


    def forward(self, feature_in, condition):
        scale = self.scale(condition)
        shift = self.shift(condition)
        feature_out = torch.mul(feature_in, (scale+1)) + shift
        return feature_out



def res_scale(vis_input_image,vis_mid_out,vis_padh,vis_padw,scale):

    _, h_up, w_up = vis_input_image.size()
    _, h_down, w_down = vis_mid_out.size()
    up_scale = h_up / h_down
    vis_down_padh = vis_padh / up_scale
    vis_down_padw = vis_padw / up_scale
    seg_h = int((h_down - vis_down_padh) + 0.5)
    seg_w = int((w_down - vis_down_padw) + 0.5)
    vis_mid_out_res = vis_mid_out[:, :seg_h, :seg_w]
    seg_h = int(seg_h / scale + 0.5)
    seg_w = int(seg_w / scale + 0.5)
    vis_mid_out_res = F.interpolate(vis_mid_out_res[None], size=(seg_h, seg_w), mode='bilinear', align_corners=False)

    return vis_mid_out_res
