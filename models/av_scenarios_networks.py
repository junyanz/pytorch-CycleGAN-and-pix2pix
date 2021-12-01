'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import functools

#########################################################################################

class PolygonEncoder(nn.Module):

    def __init__(self, opt):
        super(PolygonEncoder, self).__init__()
        self.dim_latent_polygon = opt.dim_latent_polygon
        self.kernel_size_conv_polygon = opt.kernel_size_conv_polygon
        conv1 = nn.Conv1d(in_channels=2, out_channels=self.dim_latent_polygon,
                          kernel_size=self.kernel_size_conv_polygon,
                          padding_mode='circular')
        self.conv_weights = nn.Parameter(conv1.weight)

    def forward(self, input):
        """Standard forward
        """
        # conv1d input [in_channels=2 x n_points]
        latent_polygon = nnf.conv1d(input, self.conv_weights, padding='circular')
        return latent_polygon

#########################################################################################

class MapEncoder(nn.Module):

    def __init__(self, opt, polygon_name_order):
        super(MapEncoder, self).__init__()
        self.polygon_name_order = polygon_name_order
        self.polygon_encoders = nn.ModuleList()
        for _ in self.polygon_name_order:
            self.polygon_encoders.append(PolygonEncoder(opt))
        self.dim_latent_map = opt.dim_latent_polygon * len(self.polygon_name_order)

    def forward(self, input):
        """Standard forward
        """
        pol_enc_outs = []
        for i_enc, polygon_name in enumerate(self.polygon_name_order):
            pol_enc = self.polygon_encoders[i_enc]
            pol_enc_outs.append(pol_enc(input[polygon_name]))
        out = torch.cat(pol_enc_outs)
        return out

#########################################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt, polygon_name_order):
        super(SceneGenerator, self).__init__()
        self.map_enc = MapEncoder(opt, polygon_name_order)
        self.dim_latent_scene_noise = opt.dim_latent_scene_noise
        self.batch_size = opt.batch_size
        if self.batch_size != 1:
            raise NotImplementedError

    def forward(self, map_feat):
        """Standard forward"""
        map_latent = self.map_enc(map_feat)
        latent_noise = torch.randn(self.batch_size, self.dim_latent_scene_noise)
        scene_latent= torch.concat([map_latent, latent_noise], dim=1)
        return scene_latent
#########################################################################################333

