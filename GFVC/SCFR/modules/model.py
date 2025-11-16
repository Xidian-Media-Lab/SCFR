# -*- coding: utf-8 -*-
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import *
import numpy as np
from torch.autograd import grad
from .GDN import GDN
import math
from modules.vggloss import *
from modules.dists import *
import depth

class Depth_Net(torch.nn.Module):
    def __init__(self):
        super(Depth_Net, self).__init__()
        self.depth_encoder = depth.ResnetEncoder(50, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
        loaded_dict_enc = torch.load(
            '../depth/models/depth_face_model_Voxceleb2_10w/encoder.pth',
            map_location='cpu')
        loaded_dict_dec = torch.load(
            '../depth/models/depth_face_model_Voxceleb2_10w/depth.pth',
            map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_decoder.load_state_dict(loaded_dict_dec)
        self.set_requires_grad(self.depth_encoder, False)
        self.set_requires_grad(self.depth_decoder, False)
        self.depth_decoder.eval()
        self.depth_encoder.eval()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(self, x):
        x = self.depth_decoder(self.depth_encoder(x))
        return x

class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, cross_scale_m_fusion, generator, discriminator,videocompressor, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.cross_scale_m_fusion = cross_scale_m_fusion
        self.videocompressor = videocompressor
        self.train_params = train_params
        self.scale_factor = train_params['scale_factor']
        self.scales = train_params['scales']
        self.temperature =train_params['temperature']
        self.out_channels =train_params['num_kp']       
        self.disc_scales = self.discriminator.scales
        
        self.down = AntiAliasInterpolation2d(generator.num_channels, self.scale_factor)    
            
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

        self.dists = DISTS()
        if torch.cuda.is_available():
            self.dists = self.dists.cuda()

        self.depth_net = Depth_Net()
        if torch.cuda.is_available():
            self.depth_net = self.depth_net.cuda()
             
    def forward(self, x, lambda_var):
        bs,_,width,height=x['source'].shape

        source_depth = self.depth_net(x['source'])
        driving_depth = self.depth_net(x['driving'])

        heatmap_source = self.kp_extractor(source_depth['disp', 1]) ###
        heatmap_driving = self.kp_extractor(driving_depth['disp', 1])

        heatmap_source = {'value': heatmap_source}
        heatmap_driving = {'value': heatmap_driving}

        
        lamdaloss = lambda_var
        
        if torch.cuda.is_available():                
            lambda_var = torch.tensor(lambda_var).cuda()                  
        
        total_bits_mv, quant_driving = self.videocompressor(heatmap_driving,heatmap_source)    #####

        M_i = self.cross_scale_m_fusion(heatmap_driving['value'])
        generated = self.generator(x['source'],heatmap_source=heatmap_source,heatmap_driving=heatmap_driving, M_i=M_i) #####
        generated.update({'heatmap_source': heatmap_source, 'heatmap_driving': heatmap_driving, 'depth_source': source_depth['disp', 1],
                          'depth_driving_prediction': M_i})    ####

        loss_values = {}

        pyramide_real = self.pyramid(x['driving']) 
        pyramide_generated = self.pyramid(generated['prediction'])


        ####lambda
        loss_values['lambda'] = lambda_var         

        
        ###bpp loss
        bpp_mv = total_bits_mv / (bs * width * height) #####
        loss_values['bpp'] = bpp_mv 

        ### dists loss and lpips loss
        if torch.cuda.is_available():
            prediction=prepare_image(np.transpose(generated['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]).cuda() 
            groundtruth=prepare_image(np.transpose(x['driving'].data.cpu().numpy(), [0, 2, 3, 1])[0]).cuda() 
        dists= self.dists(groundtruth,prediction, as_loss=True)
        ######
        loss_values['dists'] = dists


        
        #### rd loss optimization
        rdloss = lamdaloss*bpp_mv + dists  ###
        loss_values['rdloss'] = rdloss

        ### Perceptual Loss---Final
        if sum(self.loss_weights['perceptual_final']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_final']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_final'][i] * value
                value_total = value_total * 0.01
                loss_values['perceptual_256FINAL'] = value_total

        ### GAN Loss
        if self.loss_weights['generator_gan'] != 0:

            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)     

            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            value_total = value_total
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    value_total = value_total * 0.1
                    loss_values['feature_matching'] = value_total

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator,videocompressor, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.videocompressor= videocompressor
        
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())
        
        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)
        
        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
