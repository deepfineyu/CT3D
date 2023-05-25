
import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, stride = 1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def depth_conv2d(inp, oup, kernel=1, stride=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size = kernel, stride = stride, padding=pad, groups=inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, kernel_size=1)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, dilation=1),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class Slim(nn.Module):
    def __init__(self, n_channels, n_classes):
        
        # downsample 8

        super(Slim, self).__init__()
        
        self.num_classes = n_classes

        self.conv1 = conv_bn(n_channels, 16, 2)
        self.conv2 = conv_dw(16, 32, 1)
        self.conv3 = conv_dw(32, 32, 1) ### 2
        self.conv4 = conv_dw(32, 32, 1)
        self.conv5 = conv_dw(32, 64, 2) ### 2
        self.conv6 = conv_dw(64, 64, 1)
        self.conv7 = conv_dw(64, 64, 1)
        self.conv8 = conv_dw(64, 64, 1)

        self.conv9 = conv_dw(64, 128, 2) ### 2
        self.conv10 = conv_dw(128, 128, 1)
        self.conv11 = conv_dw(128, 128, 1)

        self.conv12 = conv_dw(128, 256, 1) ### 2
        self.conv13 = conv_dw(256, 256, 1)

        # self.conv14 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     depth_conv2d(64, 256, kernel=3, stride=2, pad=1),
        #     nn.ReLU(inplace=True)
        # )

        self.conv15 = nn.Conv2d(in_channels=256, out_channels=self.num_classes, kernel_size=1)
        self.conv16 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)



    def forward(self,inputs):

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        x = self.conv12(x)
        x = self.conv13(x)

        # x14= self.conv14(x13)
        x= self.conv15(x)
        x= self.conv16(x)

        return x

if __name__=="__main__":
    import numpy
    import time
    x = numpy.random.rand(1,1,800,800).astype("float32")
    x = torch.from_numpy(x)
    net = Slim()
    t0 = time.time()
    y = net(x)
    y = y.cpu().detach().numpy()
    t1 = time.time()
    print(t1-t0)
    print(y.shape)