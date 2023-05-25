""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_model_3d import *
from torch.nn.functional import threshold, normalize

class Up3d_noline(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')
        self.conv = DoubleConv3d(in_channels, out_channels)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
        

class Decoder3d(nn.Module):
    def __init__(self):
        super(Decoder3d, self).__init__()        
        self.numSlice = 64
        self.adapter3d = UNet3d(256, 128)     #HW (256, 64, 64, 64) -> (256, 64, 64, 64)
        self.up3d_1 = Up3d_noline(128, 64)   #HW (256, 64, 64, 64) -> (128, 64, 128, 128)
        self.up3d_2 = Up3d_noline(64, 32)   #HW (128, 64, 128, 128) -> (64, 64, 256, 256)
        self.up3d_3 = Up3d_noline(32, 3)   #HW (64, 64, 256, 256) -> (3, 64, 512, 512)

    def forward(self, embed):
        """
        embed.size() = (Batch, C, numSlice, H, W) = (Batch, 256, 64, 64,  64)
        """        
        #### image_encoder输出的embed，送入adapter3d，得到embed3后面进入mask_decoder
        embed2 = self.adapter3d(embed)
        
        #### 3d上采样
        x = self.up3d_1(embed2)
        x = self.up3d_2(x)
        x = self.up3d_3(x)
        return x

if __name__=="__main__":
    import numpy
    import torch
    unet3d = Decoder3d().cuda().half()
    embed = torch.rand(1, 256, 64, 64, 64).cuda().half()
    o = unet3d(embed)
    print(o.size())
    print(o.min(), o.max())
    
