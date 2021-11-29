'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''

import torch
import torch.nn as nn
import functools
from urban_driver.local_graph import LocalSubGraph, SinusoidalPositionalEmbedding

class MapEncoder(nn.Module):
    class MapEncoder(nn.Module):

        def __init__(self):
            super().__init__()
            self._d_local = 256
            self._subgraph_layers = 3
            self.local_subgraph = LocalSubGraph(num_layers=self._subgraph_layers, dim_in=self._d_local)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)