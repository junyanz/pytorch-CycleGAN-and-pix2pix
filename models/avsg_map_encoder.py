"""
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.avsg_moudules import MLP, PointNet

class PolygonEncoder(nn.Module):

    def __init__(self, dim_latent, n_conv_layers, kernel_size, is_closed, device):
        super(PolygonEncoder, self).__init__()
        self.device = device
        self.is_closed = is_closed
        self.dim_latent = dim_latent
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.conv_layers = []
        for i_layer in range(self.n_conv_layers):
            if i_layer == 0:
                in_channels = 2  # in the input each point has 2 channels (x,y)
            else:
                in_channels = self.dim_latent
            self.conv_layers.append(nn.Conv1d(in_channels=in_channels,
                                              out_channels=self.dim_latent,
                                              kernel_size=self.kernel_size,
                                              padding='same',
                                              padding_mode='circular',
                                              device=self.device))
        self.layers = nn.ModuleList(self.conv_layers)
        self.out_layer = nn.Linear(self.dim_latent, self.dim_latent, device=self.device)

    def forward(self, poly_points):
        """Standard forward
        input [1 x n_points  x 2 coordinates]
        """
        assert poly_points.shape[0] == 1  # assume batch_size=1

        # fit to conv1d input dimensions [batch_size=1  x in_channels=2  x n_points]
        h = torch.permute(poly_points, (0, 2, 1))

        if not self.is_closed:
            # concatenate a reflection of this sequence, to create a circular closed polygon.
            # since the model is cyclic-shift invariant, we get a pipeline that is
            # invariant to the direction of the sequence
            h = F.pad(h, (0, h.shape[2] - 1), mode='reflect').contiguous()

        # If the points sequence is too short to use the conv filter - pad in circular manner
        while h.shape[2] < self.kernel_size:
            pad_len = min(self.kernel_size - h.shape[2], h.shape[2])
            h = F.pad(h, (0, pad_len), mode='circular').contiguous()

        # We use several layers a 1d circular convolution followed by ReLu (equivariant layers)
        # and finally sum the output - this is all in all - a shift-invariant operator
        for i_layer in range(self.n_conv_layers):
            h = self.conv_layers[i_layer](h)
            h = F.relu(h)
        h = h.sum(dim=2)
        h = self.out_layer(h)
        return h


#########################################################################################


class MapEncoder(nn.Module):

    def __init__(self, opt):
        super(MapEncoder, self).__init__()
        self.device = opt.device
        self.polygon_name_order = opt.polygon_name_order
        self.closed_polygon_types = opt.closed_polygon_types
        self.dim_latent_polygon_elem = opt.dim_latent_polygon_elem
        n_polygon_types = len(opt.polygon_name_order)
        self.dim_latent_polygon_type = opt.dim_latent_polygon_type
        self.dim_latent_map = opt.dim_latent_map
        self.poly_encoder = nn.ModuleDict()
        self.sets_aggregators = nn.ModuleDict()
        for poly_type in self.polygon_name_order:
            is_closed = poly_type in self.closed_polygon_types
            self.poly_encoder[poly_type] = PolygonEncoder(dim_latent=self.dim_latent_polygon_elem,
                                                          n_conv_layers=opt.n_conv_layers_polygon,
                                                          kernel_size=opt.kernel_size_conv_polygon,
                                                          is_closed=is_closed,
                                                          device=self.device)
            self.sets_aggregators[poly_type] = PointNet(d_in=self.dim_latent_polygon_elem,
                                                        d_out=self.dim_latent_polygon_type,
                                                        d_hid=self.dim_latent_polygon_type,
                                                        n_layers=opt.n_layers_sets_aggregator,
                                                        opt=opt)
        self.poly_types_aggregator = MLP(d_in=self.dim_latent_polygon_type * n_polygon_types,
                                         d_out=self.dim_latent_map,
                                         d_hid=self.dim_latent_map,
                                         n_layers=opt.n_layers_poly_types_aggregator,
                                         opt=opt)

    def forward(self, map_feat):
        """Standard forward
        """
        latents_per_poly_type = []
        for i_poly_type, poly_type in enumerate(self.polygon_name_order):
            # Get the latent embedding of all elements of this type of polygons:
            poly_encoder = self.poly_encoder[poly_type]
            poly_elements = map_feat[poly_type]
            if len(poly_elements) == 0:
                # if there are no polygon of this type in the scene:
                latent_poly_type = torch.zeros(self.dim_latent_polygon_type, device=self.device)
            else:
                poly_latent_per_elem = []
                for poly_elem in poly_elements:
                    # Transform from sequence of points to a fixed size vector,
                    # using a circular-shift-invariant module
                    poly_elem_latent = poly_encoder(poly_elem)
                    poly_latent_per_elem.append(poly_elem_latent)
                # Run PointNet to aggregate all polygon elements of this  polygon type
                poly_latent_per_elem = torch.stack(poly_latent_per_elem)
                latent_poly_type = self.sets_aggregators[poly_type](poly_latent_per_elem)
            latents_per_poly_type.append(latent_poly_type)
        poly_types_latents = torch.cat(latents_per_poly_type)
        map_latent = self.poly_types_aggregator(poly_types_latents)
        return map_latent
