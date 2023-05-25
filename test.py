
import torch
from unet import UNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = UNet(n_channels=1, n_classes=6, bilinear=True).to(device)

x = torch.rand(1,1,512,512).to(device)

o = net(x)

print(o.shape)