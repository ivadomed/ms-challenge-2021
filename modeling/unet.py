import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils import init_weights


class DownConvBlock(nn.Module):
    """
    Each block consists of 3 conv layers with ReLU non-linear activation. A pooling layer is added b/w each block.
    """
    def __init__(self, in_dim, out_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []
        if pool:
            layers.append(nn.AvgPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=True))

        layers.append(nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        return self.layers(input)


class UpConvBlock(nn.Module):
    """
    Consists of a trilinear upsampling layer followed by a convolutional layers and then a DownConvBlock
    """
    def __init__(self, in_dim, out_dim, initializers, padding, trilinear=True):
        super(UpConvBlock, self).__init__()
        self.trilinear = trilinear
        self.conv_block = DownConvBlock(in_dim, out_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.trilinear:
            up = nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=True)

        # print(up.shape, "\t", bridge.shape)
        assert up.shape[3] == bridge.shape[3]   # checks if the first dimension of the inputs match
        out = torch.cat([up, bridge], 1)
        # print("shape after concatenation: ", out.shape)
        out = self.conv_block(out)
        # print("shape after Concat+Downblock: ", out.shape)

        return out


class Unet(nn.Module):
    """
    Implementation of the standard U-Net module.
    """
    def __init__(self, in_channels, num_classes, num_feat_maps, initializers, padding=True):
        super(Unet, self).__init__()
        # print(padding)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_feat_maps = num_feat_maps
        self.padding = padding
        self.activation_maps = []
        self.contracting_path = nn.ModuleList()

        # Contractive Path
        for i in range(len(self.num_feat_maps)):
            inp = self.in_channels if i == 0 else out
            out = self.num_feat_maps[i]
            if i == 0:
                pool = False
            else:
                pool = True
            self.contracting_path.append(DownConvBlock(inp, out, initializers, padding, pool=pool))

        # print(self.contracting_path)

        # Upsampling Path
        self.upsampling_path = nn.ModuleList()
        n = len(self.num_feat_maps) - 2
        for i in range(n, -1, -1):
            inp = out + self.num_feat_maps[i]   # sets the right no. of input channels for Concat+DownBlock
            out = self.num_feat_maps[i]     # sets the right no. of output channels for next (Concat+DownBlock)'s input
            self.upsampling_path.append(UpConvBlock(inp, out, initializers, padding))

        # print(self.upsampling_path)

    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            # print(i, down)
            x = down(x)
            # print("After DownConv: ", i, "\t", x.shape)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            # print("Before UpConv: ", x.shape, "\t", blocks[-i-1].shape)
            # print(up)
            x = up(x, blocks[-i-1])
            # print("After UpConv: ", i, x.shape)

        del blocks

        # for saving the activations
        if val:
            self.activation_maps.append(x)

        return x


if __name__ == "__main__":
    inp = torch.randn(4, 1, 128, 128, 128)
    net = Unet(in_channels=1, num_classes=1, num_feat_maps=[16, 32, 64, 128])
    out = net(inp, False)
    print(out.shape)


