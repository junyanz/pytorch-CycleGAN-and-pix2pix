'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''

import torch
import torch.nn as nn
import functools
#########################################################################################333

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.map_enc = MapEncoder()
        self.dim_latent_scene_noise = opt.dim_latent_scene_noise
        self.batch_size = opt.batch_size


    def forward(self, in_map_feat):
        """Standard forward"""
        map_latent = self.map_enc(in_map_feat)
        latent_noise = torch.randn(self.batch_size, self.dim_latent_scene_noise)
        scene_latent = torch.concat(map_latent, latent_noise, dim=1)

        return self.model(input)
#########################################################################################333


class MapEncoder(nn.Module):
    class MapEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels=2,
                                   out_channels=self.dim_latent_polygon,
                                   kernel_size=self.kernel_size_conv_polygon,
                                   padding='circular')


    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    #########################################################################################333
