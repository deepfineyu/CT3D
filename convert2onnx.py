import argparse
import logging
import os
import glob

import numpy as np
import torch

from unet import UNet
import cv2


model_path = './checkpoints/CP_epoch20.pth'
# load weight
net = UNet(n_channels=3, n_classes=5)
device = torch.device('cpu')
net.to(device=device)
f_model = torch.load(model_path, map_location=device)
net.load_state_dict(f_model)

net.eval()
print('Finished loading model!')
print(net)

inputs = torch.randn(1, 3, 400, 400).to(device)




mod = torch.jit.trace(net, inputs)
torch.jit.save(mod, "rscan_iris.pt")

##################export###############
output_onnx = 'rscan_iris.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input0"]
output_names = ["output0"]
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                                input_names=input_names, output_names=output_names, opset_version=11, keep_initializers_as_inputs=True)
##################end###############