'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''

import torch
import torch.nn as nn
import functools


class SceneGenerator(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class MapEncoder(nn.Module):
    class MapEncoder(nn.Module):

        def __init__(self):
            super().__init__()


    def forward(self, input):
        """Standard forward"""
        return self.model(input)