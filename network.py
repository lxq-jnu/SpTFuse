import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.attention import TransformerEncoderBlock,DualTransformerEncoderBlock
from models.common import MIM,SEM
from models.SEWeight import SEWeightModule as SE
import os
import matplotlib.pyplot as plt


def d_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class Decoder_2(nn.Module):
    def __init__(self, dim):
        super(Decoder_2, self).__init__()

        self.dconv_up3 = d_conv(256 + 512, 256)
        self.dconv_up2 = d_conv(128 + 256, 128)
        self.dconv_up1 = d_conv(128 + 64, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(in_channels=64, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x, add_fea, H, W, encoder_fea):
        x1 = F.interpolate(x,scale_factor=2, mode='bilinear', align_corners=False)
        x1 = torch.cat([x1, encoder_fea[3]], dim=1)

        x2 = self.dconv_up3(x1)
        x2 = x2 + add_fea[1]
        x2 = F.interpolate(x2,scale_factor=2, mode='bilinear', align_corners=False)
        x2 = torch.cat([x2, encoder_fea[2]], dim=1)

        x3 = self.dconv_up2(x2)
        x3 = x3 + add_fea[0]
        x3 = F.interpolate(x3,scale_factor=2, mode='bilinear', align_corners=False)
        x3 = torch.cat([x3, encoder_fea[1]], dim=1)

        x4 = self.dconv_up1(x3)
        x4 = F.interpolate(x4,scale_factor=2, mode='bilinear', align_corners=False)
        x4 = torch.cat([x4, encoder_fea[0]], dim=1)

        x5 = self.dconv_last(x4)#x5:C=64
        x5 = F.interpolate(x5,scale_factor=2, mode='bilinear', align_corners=True)

        x_final = nn.Tanh()(self.conv_out(x5)) / 2 + 0.5

        return x_final


class SpTFuse(nn.Module):

    def __init__(self):
        super(SpTFuse, self).__init__()
        self.num_resnet_layers = 34
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)

        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)

        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)

        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)

        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)

        self.encoder_thermal_conv1 = resnet_raw_model1.conv1
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool

        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool

        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        self.dim_in_channel_list=[64,128,256,512]
        self.seg_channel = 256
        self.sft_in_channel = [64,128,256,256]
        self.sft_out_channel = [64,128,256,512]
        self.high_fuse2 = SDFM_dual(self.dim_in_channel_list[3], self.seg_channel, self.sft_in_channel[3], self.sft_out_channel[3])
        self.high_fuse1 = SDFM_dual(self.dim_in_channel_list[2], self.seg_channel, self.sft_in_channel[2], self.sft_out_channel[2])
        self.low_fuse = SDFM_one(self.dim_in_channel_list[1], self.seg_channel, self.sft_in_channel[1], self.sft_out_channel[1])

        self.decoder1 = Decoder_2(self.dim_in_channel_list)
        self.con3x3_64 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for i in range(2)
        ])
        channel=[128,256,512]
        self.con3x3_down = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=channel[i]*2, out_channels=channel[i], kernel_size=3,padding=1),
                nn.BatchNorm2d(channel[i]),
                nn.ReLU()
            )for i in range(2)])

        self.con3x3_add =nn.Sequential(
            nn.Conv2d(in_channels=channel[2], out_channels=channel[2], kernel_size=3,padding=1),
            nn.BatchNorm2d(channel[2]),
            nn.ReLU()
            )
   

    def forward(self, rgb, depth, vis_sig, ir_seg,H,W,C):

        _,_,H_seg,W_seg = vis_sig.size()

        rgb = rgb
        thermal = depth[:, :1, ...]
        thermal = torch.cat([thermal, thermal, thermal], dim=1) 


        decoder_add_fea=[]
        encoder_fea_list=[]

        rgb = self.encoder_rgb_conv1(rgb)  #
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb)
        thermal = self.encoder_thermal_conv1(thermal)  #
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)

        encoder_fea = torch.cat([rgb,thermal],dim = 1)
        encoder_fea = self.con3x3_64[0](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        #maxpooling
        rgb = self.encoder_rgb_maxpool(rgb)
        thermal = self.encoder_thermal_maxpool(thermal)

        rgb1 = self.encoder_rgb_layer1(rgb)
        thermal1 = self.encoder_thermal_layer1(thermal)
        encoder_fea = torch.cat([rgb1,thermal1],dim = 1)
        encoder_fea = self.con3x3_64[1](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        rgb2 = self.encoder_rgb_layer2(rgb1)
        thermal2 = self.encoder_thermal_layer2(thermal1)
        encoder_fea = torch.cat([rgb2,thermal2],dim = 1)
        encoder_fea = self.con3x3_down[0](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        rgb2_sem, thermal2_sem, add_fea,vis_cross,ir_cross,vis_ploss1,ir_ploss1  = self.low_fuse(rgb2,thermal2,vis_sig,ir_seg,C)
        rgb2 = rgb2 + rgb2_sem
        thermal2 = thermal2 + thermal2_sem
        decoder_add_fea.append(add_fea)

        rgb3 = self.encoder_rgb_layer3(rgb2) 
        thermal3 = self.encoder_thermal_layer3(thermal2) 
        encoder_fea = torch.cat([rgb3,thermal3],dim = 1)
        encoder_fea = self.con3x3_down[1](encoder_fea)
        encoder_fea_list.append(encoder_fea)

        rgb3_sem, thermal3_sem, add_fea,vis_cross,ir_cross,vis_ploss2,ir_ploss2  = self.high_fuse1(rgb3,thermal3,vis_sig,ir_seg,C)
        rgb3 = rgb3 + rgb3_sem
        thermal3 = thermal3 + thermal3_sem
        decoder_add_fea.append(add_fea)
        
        rgb4= self.encoder_rgb_layer4(rgb3)
        thermal4 = self.encoder_thermal_layer4(thermal3) 

        rgb4_sem, thermal4_sem, add_fea,vis_cross,ir_cross,vis_ploss3,ir_ploss3  = self.high_fuse2(rgb4,thermal4,vis_sig,ir_seg,C)
        rgb4 = rgb4 + rgb4_sem
        thermal4 = thermal4 + thermal4_sem

        fuse_fea = rgb4 + thermal4
        fuse_fea = self.con3x3_add(fuse_fea)

        decoder_fea = self.decoder1(fuse_fea,decoder_add_fea,H,W,encoder_fea_list)

        vis_ploss_all = vis_ploss1+vis_ploss2+vis_ploss3
        ir_ploss_all = ir_ploss1+ir_ploss2+ir_ploss3

        return  decoder_fea,vis_ploss_all,ir_ploss_all

class SDFM_one(nn.Module):
    def __init__(self, dim_in=32, dim_out=256, feature_channel=32,out_channel=32, nhead=8):
        super(SDFM_one,self).__init__()

        self.encoder_block_one_1 = TransformerEncoderBlock(dim_in,dim_out, nhead)
        self.encoder_block_one_2 = TransformerEncoderBlock(dim_in, dim_out, nhead)

        self.MIM_1=MIM(feature_channel=feature_channel,out_channel=out_channel)
        self.MIM_2=MIM(feature_channel=feature_channel,out_channel=out_channel)
        self.conv = nn.Conv2d(in_channels=feature_channel * 2, out_channels=out_channel , kernel_size=3, padding=1)

        self.se = SE(out_channel)
        self.conv1x1_in = nn.Conv2d(dim_in, dim_out, kernel_size=1)


    def forward(self, rgb, ir, vis_seg, ir_seg,C):
        _,c,h,w = rgb.size()
        _,_,h_seg,w_seg = vis_seg.size()
        if h > h_seg :
            n = h // h_seg
            n = float(n)
            vis_seg = F.interpolate(vis_seg, scale_factor=n)
            ir_seg = F.interpolate(ir_seg, scale_factor=n)
        elif h < h_seg :
            n = h_seg // h
            vis_seg=F.avg_pool2d(vis_seg,kernel_size=n,stride=n)
            ir_seg=F.avg_pool2d(ir_seg,kernel_size=n,stride=n)

        vis_cross = self.encoder_block_one_1(rgb, vis_seg,C)
        ir_cross = self.encoder_block_one_2(ir, ir_seg,C)

        if c != C:
            vis_prior = self.conv1x1_in(vis_cross)
            ir_prior = self.conv1x1_in(ir_cross)
        vis_ploss = F.mse_loss(vis_prior,vis_seg)
        ir_ploss = F.mse_loss(ir_prior,ir_seg)

        vis_out = self.MIM_1(vis_cross, ir_cross)
        ir_out = self.MIM_2(ir_cross, vis_cross)

        add_fea = torch.cat([vis_cross,ir_cross],dim=1)
        add_fea = self.conv(add_fea)
        add_fea_w = self.se(add_fea) 
        add_fea = add_fea * add_fea_w

        return vis_out,ir_out,add_fea,vis_cross,ir_cross,vis_ploss,ir_ploss

class SDFM_dual(nn.Module):
    def __init__(self, dim_in=32, dim_out=256, feature_channel=32,out_channel=32, nhead=8):
        super(SDFM_dual,self).__init__()

        self.encoder_block_3 = DualTransformerEncoderBlock(dim_in,dim_out, nhead)
        self.encoder_block_4 = DualTransformerEncoderBlock(dim_in,dim_out, nhead)

        self.SEM_1 = SEM(feature_channel=feature_channel, out_channel=out_channel)
        self.SEM_2 = SEM(feature_channel=feature_channel, out_channel=out_channel)

        # self.conv1 = nn.Conv2d(in_channels=feature_channel * 2, out_channels=out_channel , kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel , kernel_size=3, padding=1)

        # self.se1 = SE(out_channel)
        self.se2 = SE(out_channel)
        self.conv1x1_in = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, rgb, ir, vis_seg, ir_seg,C):
        _,c,h,w = rgb.size()
        _,_,h_seg,w_seg = vis_seg.size()
        if h > h_seg :
            n = h // h_seg
            vis_seg = F.interpolate(vis_seg, scale_factor=float(n))
            ir_seg = F.interpolate(ir_seg, scale_factor=float(n))

        elif h < h_seg :
            n = h_seg // h
            vis_seg=F.avg_pool2d(vis_seg,kernel_size=(n,n),stride=(n,n))
            ir_seg=F.avg_pool2d(ir_seg,kernel_size=(n,n),stride=(n,n))

        vis_cross = self.encoder_block_3(rgb, vis_seg,C)
        ir_cross = self.encoder_block_4(ir, ir_seg,C)

        if c != C:
            vis_prior = self.conv1x1_in(vis_cross)
            ir_prior = self.conv1x1_in(ir_cross)
        else:
            vis_prior = vis_cross
            ir_prior = ir_cross
        vis_ploss = F.mse_loss(vis_prior,vis_seg)
        ir_ploss = F.mse_loss(ir_prior,ir_seg)

        vis_out = self.SEM_1(vis_cross, vis_seg)
        ir_out = self.SEM_2(ir_cross, ir_seg)

        add_fea = torch.cat([vis_cross, ir_cross], dim=1)
        add_fea = self.conv2(add_fea)
        add_fea_w = self.se2(add_fea) 
        add_fea = add_fea * add_fea_w

        return vis_out,ir_out,add_fea,vis_cross,ir_cross,vis_ploss,ir_ploss
