import torch
import torch.nn as nn
from module.CBAM import CBAM
from module.SENet import SELayer


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MDRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = ConvBNReLU(in_channels, out_channels, dilation=1)

        self.branch1_conv1 = ConvBNReLU(out_channels, out_channels, dilation=1)
        self.branch1_conv2 = ConvBNReLU(out_channels, out_channels, dilation=1)

        self.branch2_conv1 = ConvBNReLU(out_channels, out_channels, dilation=1)
        self.branch2_conv2 = ConvBNReLU(out_channels, out_channels, dilation=3)
        self.branch2_conv3 = ConvBNReLU(out_channels, out_channels, dilation=3)

        self.branch3_conv1 = ConvBNReLU(out_channels, out_channels, dilation=3)
        self.branch3_conv2 = ConvBNReLU(out_channels, out_channels, dilation=3)
        self.branch3_conv3 = ConvBNReLU(out_channels, out_channels, dilation=5)

        self.fuse_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)

        self.cbam = CBAM(out_channels)

        self.out_conv = ConvBNReLU(out_channels, out_channels, dilation=1)

    def forward(self, x):
        x0 = self.conv_in(x)

        b1 = self.branch1_conv1(x0)
        b1 = self.branch1_conv2(b1)

        b2 = self.branch2_conv1(x0)
        b2 = self.branch2_conv2(b2)
        b2 = self.branch2_conv3(b2)

        b3 = self.branch3_conv1(x0)
        b3 = self.branch3_conv2(b3)
        b3 = self.branch3_conv3(b3)

        multi = torch.cat([b1, b2, b3], dim=1)
        multi = self.fuse_conv(multi)
        multi = self.fuse_bn(multi)
        multi = self.se(multi)

        cbam_out = self.cbam(x0)

        out = multi + cbam_out
        out = self.out_conv(out)
        return out

