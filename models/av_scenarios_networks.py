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
        self.layer_dims = [d_in] + (n_layers - 1) * [d_hid] + [d_out]
        self.A = []
        self.B = []
        for i_layer in range(n_layers - 1):
            # each layer the function that operates on each element in the set x is
            # f(x) = ReLu(A x + B * (sum over all non x elements) )
            layer_dims = (self.layer_dims[i_layer], self.layer_dims[i_layer + 1])
            self.A.append(nn.Parameter(torch.Tensor(*layer_dims)))
            self.B.append(nn.Parameter(torch.Tensor(*layer_dims)))
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

    def __init__(self, dim_latent, n_conv_layers, kernel_size):
        super(PolygonEncoder, self).__init__()
        self.dim_latent = dim_latent
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.conv_layers = []
        for i_layer in range(self.n_conv_layers):
            self.conv_layers.append(nn.Conv1d(in_channels=2,
                                              out_channels=self.dim_latent,
                                              kernel_size=self.kernel_size,
                                              padding_mode='circular'))
        self.layers = nn.ModuleList(self.conv_layers)

    def forward(self, input):
        """Standard forward
        input [1 x n_points  x 2 coordinates]
        """

        # fit to conv1d input dimensions [1  x in_channels=2  x n_points]
        h = torch.permute(input, (0, 2, 1))

        # We several layers a 1d circular convolution followed by ReLu (equivariant layers)
        # and finally sum the output - this is all in all - a shift-invariant operator
        for i_layer in range(self.n_conv_layers):
            h = self.conv1(h)
            h = nn.ReLU(h)
        return h.sum(dim=2)


#########################################################################################

class MapEncoder(nn.Module):

    def __init__(self, opt):
        super(MapEncoder, self).__init__()
        self.polygon_name_order = opt.polygon_name_order
        self.dim_latent_polygon_elem = opt.dim_latent_polygon_elem
        n_polygon_types = len(opt.polygon_name_order)
        self.n_poinnet_layers = 3
        self.dim_latent_polygon_type = opt.dim_latent_polygon_type
        self.dim_latent_map = opt.dim_latent_map
        self.poly_encoderoders = nn.ModuleList()
        self.elements_aggregators = nn.ModuleList()
        for _ in self.polygon_name_order:
            self.poly_encoder.append(
                PolygonEncoder(dim_latent=self.dim_latent_polygon_elem,
                               n_conv_layers=opt.n_conv_layers_polygon,
                               kernel_size=opt.kernel_size_conv_polygon))
            self.elements_aggregators.append(PointNet(n_layers=self.n_poinnet_layers,
                                            d_in=self.dim_latent_polygon_elem,
                                            d_out=self.dim_latent_polygon_type,
                                            d_hid=self.dim_latent_polygon_type))
        self.poly_types_aggregator =\
            torch.nn.Linear(in_features=self.dim_latent_polygon_type * n_polygon_types,
                            out_features=self.dim_latent_map)


    def forward(self, input):
        """Standard forward
        """
        poly_types_latents = []
        for i_poly_type, poly_type_name in enumerate(self.polygon_name_order):
            # Get the latent embedding of all elements of this type of polygons:
            poly_encoder = self.poly_encoder[i_poly_type]
            poly_elements = input[poly_type_name]
            poly_latent_per_elem = []
            for poly_elem in poly_elements:
                # Transform from sequence of points to a fixed size vector,
                # using a a circular-shift-invariant module
                poly_latent = poly_encoder(poly_elem)
                poly_latent_per_elem.append(poly_latent)
            # Run PointNet to aggregate all polygon elements of this  polygon type
            poly_latent_per_elem = torch.stack(poly_latent_per_elem)
            elments_agg = self.elements_aggregators[i_poly_type]
            poly_types_latents.append(elments_agg(poly_latent_per_elem))
        poly_types_latents = torch.stack(poly_types_latents)
        map_latent = self.poly_types_aggregator(poly_types_latents)
        return map_latent


#########################################################################################

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.map_enc = MapEncoder(opt)
        # Debug - print parameter names: print([a[0] for a in self.named_parameters()])
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
