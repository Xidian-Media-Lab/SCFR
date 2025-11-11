from torch import nn

import torch.nn.functional as F
import torch
import numpy as np
from torchvision import transforms
import math
import torch.nn.utils.spectral_norm as spectral_norm

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type())
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    #print(meshed.shape)
    return meshed

class UpBlock2d_densemotion(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d_densemotion, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d_densemotion(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d_densemotion, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm3 = nn.InstanceNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + x
        out_ = self.norm3(out)
        out_ = F.relu(out_)
        out_ = self.conv3(out_)
        out = out + out_
        return out

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class DHFE_UpBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(DHFE_UpBlock2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = 2
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels * self.reduction, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.Conv2d(self.in_channels * self.reduction, self.in_channels, kernel_size=kernel_size, stride=1,
                      padding=1, groups=self.in_channels, bias=False),
            nn.InstanceNorm2d(self.in_channels, affine=True),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class DHFE_DownBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, activation='relu'):
        super(DHFE_DownBlock2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = 2
        self.down_dwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.reduction, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.Conv2d(self.in_channels * self.reduction, self.in_channels, kernel_size=kernel_size, stride=1,
                      padding=1, groups=self.in_channels, bias=False),
            nn.InstanceNorm2d(self.in_channels, affine=True),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.down_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, act=nn.LeakyReLU(negative_slope=0.1), stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        identity = x

        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))

        out = out + identity
        # out = self.act(out)

        return out
class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # nn.functional.linear()
        out = normalized * (1 + gamma) + beta
        return out
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
class DHFE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(DHFE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = 2
        self.down_dwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.reduction, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=self.in_channels, bias=False),
            nn.Conv2d(self.in_channels * self.reduction, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=self.in_channels, bias=False),
            nn.InstanceNorm2d(self.in_channels, affine=True),
            act_layer(activation, inplace=True),
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ = self.down_dwc(x)
        x_ = channel_shuffle(x_, self.in_channels)
        x = self.pwc(x_)
        return x

class UpBlock2d_dense_motion(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d_dense_motion, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d_dense_motion(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d_dense_motion, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out

def get_parameter_groups(
    model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None
):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or name.endswith(".gamma")
            or name.endswith(".beta")
        ):
            group_name = "no_decay"
            this_weight_decay = 0.0
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.0

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DHFE_DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1)))))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(DHFE_UpBlock2d(in_filters, out_filters))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        
#         if torch.cuda.is_available():  ##
#             out = out.cuda()  ###
            
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


# +
def abs(x):
    return torch.sqrt(x[:,:,:,:,0]**2+x[:,:,:,:,1]**2+1e-12)

def real(x):
    return x[:,:,:,:,0]

def imag(x):
    return x[:,:,:,:,1]

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def preprocess_lab(lab):
		L_chan, a_chan, b_chan =torch.unbind(lab,dim=2)
		# L_chan: black and white with input range [0, 100]
		# a_chan/b_chan: color channels with input range ~[-110, 110], not exact
		# [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
		return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]

def deprocess_lab(L_chan, a_chan, b_chan):
		#TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
			   # ( we process individual images but deprocess batches)
		#return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
		return torch.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2)

def rgb_to_lab(srgb):
    srgb = srgb/255
    srgb_pixels = torch.reshape(srgb, [-1, 3])
    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
	
    rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor).to(device)
	
    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).to(device))

    epsilon = 6.0/29.0
    linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).to(device)
    exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).to(device)
    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [  0.0,  500.0,    0.0], # fx
        [116.0, -500.0,  200.0], # fy
        [  0.0,    0.0, -200.0], # fz
    ]).type(torch.FloatTensor).to(device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device)
    #return tf.reshape(lab_pixels, tf.shape(srgb))
    return torch.reshape(lab_pixels, srgb.shape)

def lab_to_rgb(lab):
		lab_pixels = torch.reshape(lab, [-1, 3])
		# convert to fxfyfz
		lab_to_fxfyfz = torch.tensor([
			#   fx      fy        fz
			[1/116.0, 1/116.0,  1/116.0], # l
			[1/500.0,     0.0,      0.0], # a
			[    0.0,     0.0, -1/200.0], # b
		]).type(torch.FloatTensor).to(device)
		fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device), lab_to_fxfyfz)

		# convert to xyz
		epsilon = 6.0/29.0
		linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
		exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)


		xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

		# denormalize for D65 white point
		xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device))


		xyz_to_rgb = torch.tensor([
			#     r           g          b
			[ 3.2404542, -0.9692660,  0.0556434], # x
			[-1.5371385,  1.8760108, -0.2040259], # y
			[-0.4985314,  0.0415560,  1.0572252], # z
		]).type(torch.FloatTensor).to(device)

		rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
		# avoid a slightly negative number messing up the conversion
		#clip
		rgb_pixels[rgb_pixels > 1] = 1
		rgb_pixels[rgb_pixels < 0] = 0

		linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
		exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
		srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
	
		return torch.reshape(srgb_pixels, lab.shape)

def spatial_normalize(x):
    min_v = torch.min(x.view(x.shape[0],1,-1),dim=2)[0]
    range_v = torch.max(x.view(x.shape[0],1,-1),dim=2)[0] - min_v
    return (x - min_v.unsqueeze(2).unsqueeze(3)) / (range_v.unsqueeze(2).unsqueeze(3)+1e-12)

def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)



def downsample(img1, img2, maxSize = 256):  ###256
    _,channels,H,W = img1.shape
    #channels,H,W = img1.shape
    f = int(max(1,np.round(min(H,W)/maxSize)))
    if f>1:
        aveKernel = (torch.ones(channels,1,f,f)/f**2).to(img1.device)
        img1 = F.conv2d(img1, aveKernel, stride=f, padding = 0, groups = channels)
        img2 = F.conv2d(img2, aveKernel, stride=f, padding = 0, groups = channels)
    return img1, img2

def extract_patches_2d(img, patch_shape=[64, 64], step=[27,27], batch_first=True, keep_last_patch=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0) and keep_last_patch:
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0) and keep_last_patch:
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches.reshape(-1,3,patch_H,patch_W)

def prepare_image(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
