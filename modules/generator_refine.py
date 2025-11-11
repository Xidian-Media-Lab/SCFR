import torch.nn as nn
from modules.util import *

class generator_refine(nn.Module):
    def __init__(self):
        super(generator_refine, self).__init__()

        self.refine = SPADEGenerator(512, 64, 2, 8)

    def forward(self, driving):
        out = self.refine(driving)
        return out

class SPADEGenerator(nn.Module):
    def __init__(self, max_features, block_expansion, num_down_blocks, num_bottleneck_blocks):
        super().__init__()
        ic = 256
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, ic)
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(DHFE_UpBlock2d(in_features, out_features))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.conv_img = nn.Conv2d(out_features, 3, kernel_size=(7, 7), padding=(3, 3))

    def forward(self, driving):
        x = self.bottleneck(driving)
        for i in range(len(self.up_blocks)):
            x = self.up_blocks[i](x)

        x = self.conv_img(x)
        x = F.sigmoid(x)

        return x
