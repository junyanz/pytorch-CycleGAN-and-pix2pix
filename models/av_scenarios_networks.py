'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import functools

#########################################################################################

class MapEncoder(nn.Module):

    def __init__(self, opt):
        super(MapEncoder, self).__init__()
        self.dim_latent_polygon = opt.dim_latent_polygon
        self.kernel_size_conv_polygon = opt.kernel_size_conv_polygon

        conv1 = nn.Conv1d(in_channels=2, out_channels=self.dim_latent_polygon,
                          kernel_size=self.kernel_size_conv_polygon,
                          padding_mode='circular')
        self.conv_weights = nn.Parameter(conv1.weight)


    def forward(self, input):
        """Standard forward
        input [batch_size x in_channels=2, n_points]
        """
        latent_polygon = nnf.conv1d(input, self.conv_weights, padding='circular')
        return latent_polygon
#########################################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.map_enc = MapEncoder(opt)
        self.dim_latent_scene_noise = opt.dim_latent_scene_noise
        self.batch_size = opt.batch_size


    def forward(self, in_map_feat):
        """Standard forward"""
        map_latent = self.map_enc(in_map_feat)
        latent_noise = torch.randn(self.batch_size, self.dim_latent_scene_noise)
        scene_latent = torch.concat(map_latent, latent_noise, dim=1)

        return map_latent
#########################################################################################333

