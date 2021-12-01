'''
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import functools

#########################################################################################
class PointNet(nn.Module):

    def __init__(self, n_layers, d_in, d_out, d_hid):
        super(PointNet, self).__init__()
        self.n_layers = n_layers
        self .layer_dims = [d_in] + (n_layers - 1 ) * [d_hid] + [d_out]
        self.A = []
        self.B = []
        for i_layer in range(n_layers - 1):
            # each layer the function that operates on each element in the set x is
            # f(x) = ReLu(A x + B * (sum over all non x elements) )
            layer_dims = (self.layer_dims[i_layer], self.layer_dims[i_layer + 1])
            self.A[i_layer] = nn.Parameter(torch.Tensor(*layer_dims))
            self.B[i_layer] = nn.Parameter(torch.Tensor(*layer_dims))
            # PyTorch's default initialization:
            nn.init.kaiming_uniform_(self.A[i_layer], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B[i_layer], a=math.sqrt(5))
        self.A_out = nn.Parameter(torch.Tensor(d_hid, d_out))
        nn.init.kaiming_uniform_(self.A_out, a=math.sqrt(5))
    def forward(self, input):
        ''''
             each layer the function that operates on each element in the set x is
            f(x) = ReLu(A x + B * (sum over all non x elements) )
            where A and B are the same for all elements, and are layer dependent.
            After that the elements are aggregated by max-pool
             and finally  a linear layer gives the output

            input is a tensor of size [num_set_elements x elem_dim]

        '''
        h = input
        n_elements = input.shape[0]
        for i_layer in range(self.n_layers - 1):
            A = self.A[i_layer]
            B = self.B[i_layer]
            pre_layer_sum = h.sum(dim=0)
            B * pre_layer_sum
            for i_elem in range(n_elements):
                sum_without_elem = pre_layer_sum - h[i_elem]
                h[i_elem] = A @ h[i_elem] + B @ sum_without_elem
            h = nn.ReLU(h)
        # apply permutation invariant aggregation over all elements
        # max-pooling in our case
        h = torch.max(h, dim=0)
        h = self.A_out * h
        return h





#########################################################################################

class PolygonEncoder(nn.Module):

    def __init__(self, opt):
        # TODO: apply several layers with ReLU in between
        super(PolygonEncoder, self).__init__()
        self.dim_latent_polygon = opt.dim_latent_polygon
        self.kernel_size_conv_polygon = opt.kernel_size_conv_polygon
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=self.dim_latent_polygon,
                               kernel_size=self.kernel_size_conv_polygon,
                               padding_mode='circular')
        self.layers = nn.ModuleList([self.conv1])

    def forward(self, input):
        """Standard forward
        input [1 x n_points  x 2 coordinates]
        """

        # fit to conv1d input dimensions [1  x in_channels=2  x n_points]
        input = torch.permute(input, (0, 2, 1))
        # We take a 1d circular convolution and sum its output - this is a shift-invariant operator
        latent_polygon = self.conv1(input).sum(dim=2)
        return latent_polygon


#########################################################################################

class MapEncoder(nn.Module):

    def __init__(self, opt):
        super(MapEncoder, self).__init__()
        self.polygon_name_order = opt.polygon_name_order
        self.n_poinnet_layers = 3
        self.polygon_encoders = nn.ModuleList()
        for _ in self.polygon_name_order:
            self.polygon_encoders.append(PolygonEncoder(opt))

        self.dim_latent_map = opt.dim_latent_polygon * len(self.polygon_name_order)

    def forward(self, input):
        """Standard forward
        """
        out = []
        for i_enc, polygon_name in enumerate(self.polygon_name_order):
            # Get the latent embedding of all elements of this type of polygons:
            pol_enc = self.polygon_encoders[i_enc]
            polygons = input[polygon_name]
            pol_latents = []
            for polygpn_elem in polygons:
                pol_latents.append(pol_enc(polygpn_elem))
            # Run PointNet on all elements of this type of polygons:
        out = torch.cat(pol_latents)
        return out


#########################################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.map_enc = MapEncoder(opt)
        self.dim_latent_scene_noise = opt.dim_latent_scene_noise
        self.batch_size = opt.batch_size
        if self.batch_size != 1:
            raise NotImplementedError

    def forward(self, map_feat):
        """Standard forward"""
        map_latent = self.map_enc(map_feat)
        latent_noise = torch.randn(self.batch_size, self.dim_latent_scene_noise)
        scene_latent = torch.concat([map_latent, latent_noise], dim=1)
        return scene_latent
#########################################################################################333
