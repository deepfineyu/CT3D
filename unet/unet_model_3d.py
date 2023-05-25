""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts_3d import *


class UNet3d(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 256):
        super(UNet3d, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv3d(n_channels, 16 )
        self.down1 = Down3d(16 , 32)
        self.down2 = Down3d(32 , 64)
        self.down3 = Down3d(64 , 128)
        self.up1 = Up3d(128 , 64)
        self.up2 = Up3d(64 , 32)
        self.up3 = Up3d(32 , 16)
        self.outc = OutConv3d(16 , n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

if __name__=="__main__":
    import numpy
    import torch
    unet3d = UNet3d()
    x = torch.rand(1,256,64,64,64)
    o = unet3d(x)
    print(o.size())