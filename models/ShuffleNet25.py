import os
import sys
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from math import log
from .butterfly import Fusion
import torch.nn.functional as F
import math
import numpy as np

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class TanhExp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(torch.exp(x))

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()

        reduced_channels_num = (in_channels // reduction_ratio) if (in_channels > reduction_ratio) else 1
        pointwise_in = nn.Conv2d(kernel_size=1,
                                 in_channels=in_channels,
                                 out_channels=reduced_channels_num)
        pointwise_out = nn.Conv2d(kernel_size=1,
                                 in_channels=reduced_channels_num,
                                 out_channels=in_channels)
        self.MLP = nn.Sequential(
            pointwise_in,
            nn.ReLU(),
            pointwise_out,
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_tensor):
        h, w = input_tensor.size(2), input_tensor.size(3)

        # Get (channels, 1, 1) tensor after MaxPool
        max_feat = F.max_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))
        # Get (channels, 1, 1) tensor after AvgPool
        avg_feat = F.avg_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))
        # Throw maxpooled and avgpooled features into shared MLP
        max_feat_mlp = self.MLP(max_feat)
        avg_feat_mlp = self.MLP(avg_feat)
        # Get channel attention map of elementwise features sum.
        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)
        return channel_attention_map

class SAM(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.ks = ks
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1)

    def _get_padding(self,
                     dim_in,
                     kernel_size,
                     stride):

        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2
        return padding

    def forward(self, input_tensor):
        c, h, w = input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)

        permuted = input_tensor.view(-1, c, h * w).permute(0,2,1)
        max_feat = F.max_pool1d(permuted, kernel_size=c, stride=c)
        max_feat = max_feat.permute(0,2,1).view(-1, 1, h, w)

        avg_feat = F.avg_pool1d(permuted, kernel_size=c, stride=c)
        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, h, w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)
        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)
        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)
        self.conv.padding = (h_pad, w_pad)
        spatial_attention_map = self.sigmoid(
            self.conv(concatenated)
        )
        return spatial_attention_map

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.CAM = CAM(in_channels)
        self.SAM = SAM()

    def forward(self, input_tensor):
        channel_att_map = self.CAM(input_tensor)
        gated_tensor = torch.mul(input_tensor, channel_att_map)
        spatial_att_map = self.SAM(gated_tensor)
        refined_tensor = torch.mul(gated_tensor, spatial_att_map) 
        return refined_tensor

class SqEx(nn.Module):
    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class FReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)
  
def conv3x3(in_channels, out_channels, stride, padding=1, groups=1):
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                    kernel_size=3, stride=stride, padding=padding,
                    groups=groups,
                    bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

class ShufflenetUnit(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ShufflenetUnit, self).__init__()
        self.downsample = downsample

        if not self.downsample:
            inplanes = inplanes // 2
            planes = planes // 2
 
        self.conv1x1_1 = conv1x1(in_channels=inplanes, out_channels=planes)
        self.conv1x1_1_bn = nn.BatchNorm2d(planes)

        self.dwconv3x3 = conv3x3(in_channels=planes, out_channels=planes, stride=stride, groups=planes)
        self.dwconv3x3_bn= nn.BatchNorm2d(planes)

        self.conv1x1_2 = conv1x1(in_channels=planes, out_channels=planes)
        # self.conv1x1_2 = GhostModule(planes, planes, kernel_size=1, stride=1)
        self.conv1x1_2_bn = nn.BatchNorm2d(planes)

        # self.relu = FReLU(in_channels=planes)
        self.relu = nn.ReLU(True)

    def _channel_split(self, features, ratio=0.5):
        """
        ratio: c'/c, default value is 0.5
        """ 
        size = features.size()[1]
        split_idx = int(size * ratio)
        return features[:,:split_idx], features[:,split_idx:]

    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1] 
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1).cuda()
        features = features[:, index]
        return features

    def forward(self, x):
        if  self.downsample:
            x1 = x
            x2 = x
        else:
            x1, x2 = self._channel_split(x)

        #----right branch----- 
        x2 = self.conv1x1_1(x2)
        x2 = self.conv1x1_1_bn(x2)
        x2 = self.relu(x2)
         
        x2 = self.dwconv3x3(x2)
        x2 = self.dwconv3x3_bn(x2)
    
        x2 = self.conv1x1_2(x2)
        x2 = self.conv1x1_2_bn(x2)
        x2 = self.relu(x2)

        #---left branch-------
        if self.downsample:
            x1 = self.downsample(x1)

        x = torch.cat([x1, x2], 1)
        x = self._channel_shuffle(x)
        return x

class ShuffleNet25(nn.Module):
    def __init__(self, feature_dim, layers_num, num_classes=2):
        super(ShuffleNet25, self).__init__()
        dim1, dim2, dim3, dim4, dim5 = feature_dim
        self.conv1      = conv3x3(in_channels=3, out_channels=dim1, stride=2, padding=1)
        # self.conv1      = GhostModule(3, dim1, kernel_size=3, stride=2)
        # self.bt         = Fusion(dim1, dim1, fusion_method="butterfly")
        self.bn1        = nn.BatchNorm2d(dim1)
        self.relu       = FReLU(dim1)
        # self.relu       = nn.ReLU(True)
        self.maxpool    = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.gm         = GhostModule(dim1, dim1, kernel_size=1, stride=1)
        
        self.stage2     = self._make_layer(dim1, dim2, layers_num[0])
        self.stage3     = self._make_layer(dim2, dim3, layers_num[1])
        self.stage4     = self._make_layer(dim3, dim4, layers_num[2])

        
        # self.cbam       = CBAM(dim4)
        # self.conv5      = conv1x1(in_channels=dim4, out_channels=dim5)
        self.conv5      = GhostModule(dim4, dim5, kernel_size=1, stride=1)
        self.sqex       = SqEx(dim5)
        self.globalpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc         = nn.Linear(dim5, num_classes)

    def _make_layer(self, dim1, dim2, blocks_num):
        half_channel = dim2 // 2
        downsample = nn.Sequential(
            conv3x3(in_channels=dim1, out_channels=dim1, stride=2, padding=1, groups=dim1),
            nn.BatchNorm2d(dim1),
            conv1x1(in_channels=dim1, out_channels=half_channel),
            nn.BatchNorm2d(half_channel),
            # FReLU(half_channel)
            nn.ReLU(True)
        )

        layers = []
        layers.append(ShufflenetUnit(dim1, half_channel, stride=2, downsample=downsample))
        for i in range(blocks_num):
            layers.append(ShufflenetUnit(dim2, dim2, stride=1))

        return nn.Sequential(*layers) 

    def forward(self,x):
        x = self.conv1(x)
        # x = self.bt(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  
        # x = self.gm(x) 
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # x = self.cbam(x)
        
        x = self.conv5(x)
        x = self.sqex(x)
        x = self.globalpool(x)

        x = x.view(-1, 1024)
        x = self.fc(x)

        return x