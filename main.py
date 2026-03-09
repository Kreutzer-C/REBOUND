import os
import numpy as np
import torch

from networks import CSANet
from utils import load_config_as_namespace

# config = load_config_as_namespace("./networks/R50_ViTB16_config.json")
# model = CSANet(config).cuda()
# model.load_from(weights=np.load(config.pretrained_path))
# model.eval()
# print("Load pretrained model successfully")

# x_sample = torch.randn(1, 1, 256, 256).cuda()
# output = model(x_sample, x_sample, x_sample)
# print(output.shape)

image = np.load("/opt/data/private/REBOUND/datasets/ABDOMINAL/processed_DDFP/CHAOST2/slices/vol_0001_slice_0013.npz")['img']
print(image)
print(image.min(), image.max())
print(image.shape)