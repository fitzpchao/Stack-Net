from model_unet import UNet
from model_deeplabv3plus import DeepLabv3_plus
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class StackNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(StackNet, self).__init__()
        self.net1=DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=False, _print=True)
        self.net2=UNet(n_channels=int(n_channels + 1),n_classes=1)
    def forward(self, x):
        out_net1 = F.sigmoid(self.net1(x))
        x = torch.cat([x, out_net1], dim=1)
        out_net2 = self.net2(x)
        out = F.sigmoid(out_net2)
        return out