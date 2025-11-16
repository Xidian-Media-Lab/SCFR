# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn.functional as F
from GFVC.SCFR.modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d,SameBlock2d,DHFE_DownBlock2d,DHFE, DHFE_UpBlock2d
from .GDN import GDN
import math
import numpy as np
import cv2
import GFVC.SCFR.depth as depth
from einops.layers.torch import Rearrange

class Encoder(torch.nn.Module):
    def __init__(self, ratio=10, block_size=32):
        super(Encoder, self).__init__()
        num_kp = 1
        self.ratio = ratio
        self.block_size = block_size
        A = torch.from_numpy(self.load_sampling_matrix()).float()
        self.A = nn.Parameter(
            Rearrange("m (1 b1 b2) -> m 1 b1 b2", b1=self.block_size)(A),
            requires_grad=True,
        )
        self.conv1 = DHFE(num_kp * 102, num_kp * 51)
        self.gdn1 = GDN(num_kp * 51)
        self.conv2 = DHFE(num_kp * 51, num_kp * 25)
        self.gdn2 = GDN(num_kp * 25)
        self.conv3 = DHFE_DownBlock2d(num_kp * 25, num_kp * 4)
        self.gdn3 = GDN(num_kp * 4)
        self.conv4 = DHFE_DownBlock2d(num_kp * 4, num_kp)
        self.gdn4 = GDN(num_kp)
        self.conv_concat = DHFE(4, 1)
    def load_sampling_matrix(self):
        path = "../sampling_matrix"
        data = np.load(f"{path}/{self.ratio}_{self.block_size}.npy")
        return data

    def forward(self, x, x_depth):
        m = self.conv_concat(torch.cat([x, x_depth], dim=1))
        m = F.conv2d(m, self.A, stride=self.block_size, padding=0, bias=None)
        m = self.gdn1(self.conv1(m))
        m = self.gdn2(self.conv2(m))
        m = self.gdn3(self.conv3(m))
        m = self.gdn4(self.conv4(m))

        return m

