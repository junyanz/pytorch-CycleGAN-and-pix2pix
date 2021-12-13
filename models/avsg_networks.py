"""
Inspired by
https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py

"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA


class MLP(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers, device):
        super(MLP, self).__init__()
        assert n_layers >= 1
        self.device = device
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid
        self.n_layers = n_layers
        layer_dims = [d_in] + (n_layers - 1) * [d_hid] + [d_out]
        modules_list = []
        for i_layer in range(n_layers - 1):
            layer_d_in = layer_dims[i_layer]
            layer_d_out = layer_dims[i_layer + 1]
            modules_list.append(nn.Linear(layer_d_in, layer_d_out, device=self.device))
            modules_list.append(nn.LayerNorm(layer_d_out, device=self.device))
            modules_list.append(nn.ReLU())
        modules_list.append(nn.Linear(layer_dims[-2], d_out, device=self.device))
        self.net = nn.Sequential(*modules_list)
        self.layer_dims = layer_dims

    def forward(self, in_vec):
        return self.net(in_vec)


#########################################################################################


class PointNet(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers, device):
        super(PointNet, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid
        self.layer_dims = [d_in] + (n_layers - 1) * [d_hid] + [d_out]
        self.matA = {}
        self.matB = {}
        for i_layer in range(n_layers - 1):
            # each layer the function that operates on each element in the set x is
            # f(x) = ReLu(A x + B * (sum over all non x elements) )
            layer_dims = (self.layer_dims[i_layer + 1], self.layer_dims[i_layer])
            self.matA[i_layer] = nn.Parameter(torch.zeros(layer_dims, device=device, requires_grad=True))
            self.matB[i_layer] = nn.Parameter(torch.zeros(layer_dims, device=device, requires_grad=True))
            self.register_parameter(name=f'matA_{i_layer}', param=self.matA[i_layer])
            self.register_parameter(name=f'matB_{i_layer}', param=self.matB[i_layer])
            # PyTorch's default initialization:
            nn.init.kaiming_uniform_(self.matA[i_layer], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.matB[i_layer], a=math.sqrt(5))
        self.out_layer = nn.Linear(d_hid, d_out, device=self.device)
        self.layer_normalizer = nn.LayerNorm(d_hid, device=self.device)

    def forward(self, in_set):
        """'
             each layer the function that operates on each element in the set x is
            f(x) = ReLu(A x + B * (sum over all non x elements) )
            where A and B are the same for all elements, and are layer dependent.
            After that the elements are aggregated by max-pool
             and finally  a linear layer gives the output

            input is a tensor of size [num_set_elements x elem_dim]

        """
        h = in_set
        n_elements = in_set.shape[0]
        for i_layer in range(self.n_layers - 1):
            matA = self.matA[i_layer]
            matB = self.matB[i_layer]
            pre_layer_sum = h.sum(dim=0).squeeze()
            h_new_lst = []
            for i_elem in range(n_elements):
                h_elem = h[i_elem].squeeze()
                sum_without_elem = pre_layer_sum - h_elem
                h_new = matA @ h_elem + matB @ sum_without_elem
                h_new_lst.append(h_new)
            h = torch.stack(h_new_lst)
            h = self.layer_normalizer(h)
            h = F.relu(h)
        # apply permutation invariant aggregation over all elements
        # max-pooling in our case
        h = torch.max(h, dim=0).values
        h = self.out_layer(h)
        return h


#########################################################################################


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
        self.n_point_net_layers = 3
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
                                                        n_layers=self.n_point_net_layers,
                                                        device=self.device)
        self.poly_types_aggregator = MLP(d_in=self.dim_latent_polygon_type * n_polygon_types,
                                         d_out=self.dim_latent_map,
                                         d_hid=self.dim_latent_map,
                                         n_layers=3,
                                         device=self.device)

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


#########################################################################################
def project_to_agent_feat(raw_vec):
    # Project the generator output to the feature vectors domain:
    agent_feat = torch.cat([
        # Coordinates 0,1 are centroid x,y - no need to project
        raw_vec[0:2],
        # Coordinates 2,3 are yaw_cos, yaw_sin - project to unit circle
        raw_vec[2:4] / LA.vector_norm(raw_vec[2:4], ord=2),
        # Coordinates 4,5,6 are extent_length, extent_width, speed project to positive numbers
        F.softplus(raw_vec[4:7]),
        # Coordinates 7,8,9 are one-hot vector - project to 3-simplex
        F.softmax(raw_vec[7:10], dim=0)
    ])
    return agent_feat


#########################################################################################

class DecoderUnit(nn.Module):

    def __init__(self, opt, dim_context, dim_out):
        super(DecoderUnit, self).__init__()
        dim_hid = dim_context
        self.device = opt.device
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.gru = nn.GRUCell(dim_hid, dim_hid)
        self.input_mlp = MLP(d_in=dim_out + dim_hid,
                             d_out=dim_hid,
                             d_hid=dim_hid,
                             n_layers=opt.gru_in_layers,
                             device=self.device)
        self.out_mlp = MLP(d_in=dim_hid,
                           d_out=dim_out,
                           d_hid=dim_hid,
                           n_layers=opt.gru_out_layers,
                           device=self.device)
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        assert self.agent_feat_vec_coord_labels == ['centroid_x', 'centroid_y', 'yaw_cos', 'yaw_sin',
                                                    'extent_length', 'extent_width', 'speed',
                                                    'is_CAR', 'is_CYCLIST', 'is_PEDESTRIAN']

    def forward(self, context_vec, prev_hidden, attn_scores, prev_agent_feat):
        # the input layer takes in the attention-applied context concatenated with the previous out features
        attn_weights = F.softmax(attn_scores, dim=0)
        attn_applied = attn_weights * context_vec
        gru_input = torch.cat([attn_applied, prev_agent_feat])
        gru_input = self.input_mlp(gru_input)
        gru_input = F.relu(gru_input)
        hidden = self.gru(gru_input.unsqueeze(0), prev_hidden.unsqueeze(0))
        hidden = hidden[0]
        output_feat = self.out_mlp(hidden)

        # Project the generator output to the feature vectors domain:
        agent_feat = project_to_agent_feat(output_feat)
        return agent_feat, hidden


##############################################################################################

class AgentsDecoderGRU(nn.Module):
    # based on:
    # * Show, Attend and Tell: Neural Image Caption Generation with Visual Attention  https://arxiv.org/abs/1502.03044\
    # * https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
    # * https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    # * https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa

    def __init__(self, opt, device):
        super(AgentsDecoderGRU, self).__init__()
        self.device = device
        self.dim_latent_scene = opt.dim_latent_scene
        self.dim_agents_decoder_hid = opt.dim_agents_decoder_hid
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.num_agents = opt.num_agents
        self.decoder_unit = DecoderUnit(opt,
                                        dim_context=self.dim_latent_scene,
                                        dim_out=self.dim_agent_feat_vec)

    def forward(self, scene_latent, n_agents):
        prev_hidden = scene_latent
        attn_scores = torch.ones_like(prev_hidden)
        prev_agent_feat = torch.zeros(self.dim_agent_feat_vec, device=self.device)
        agents_feat_vec_list = []
        for i_agent in range(n_agents):
            agent_feat, next_hidden = self.decoder_unit(
                context_vec=scene_latent,
                prev_hidden=prev_hidden,
                attn_scores=attn_scores,
                prev_agent_feat=prev_agent_feat)
            prev_hidden = next_hidden
            attn_scores = next_hidden
            prev_agent_feat = agent_feat
            agents_feat_vec_list.append(agent_feat)
        return agents_feat_vec_list


# # Sample hard categorical using "Straight-through" , returns one-hot vector
# stop_flag = F.gumbel_softmax(logits=stop_score, tau=1, hard=True)
# if i_agent > 0 and stop_flag > 0.5:
#     # Stop flag is ignored at i=0, since we want at least one agent (including the AV)  in the scene
#     break
# else:
#########################################################################################
##############################################################################################

class AgentsDecoderMLP(nn.Module):

    def __init__(self, opt, device):
        super(AgentsDecoderMLP, self).__init__()
        self.device = device
        self.dim_latent_scene = opt.dim_latent_scene
        self.dim_agents_decoder_hid = opt.dim_agents_decoder_hid
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.num_agents = opt.num_agents

        self.decoder = MLP(d_in=self.dim_latent_scene,
                           d_out=self.dim_agent_feat_vec * self.num_agents,
                           d_hid=self.dim_agents_decoder_hid,
                           n_layers=4,
                           device=self.device)

    def forward(self, scene_latent, n_agents):
        assert n_agents == self.num_agents
        out_vec = self.decoder(scene_latent)

        agents_feat_vec_list = []
        for i_agent in range(n_agents):
            output_feat = out_vec[i_agent * self.dim_agent_feat_vec:(i_agent + 1) * self.dim_agent_feat_vec]
            # Project the generator output to the feature vectors domain:
            agent_feat = project_to_agent_feat(output_feat)
            agents_feat_vec_list.append(agent_feat)
        return agents_feat_vec_list


#########

class SceneGenerator(nn.Module):

    def __init__(self, opt):
        super(SceneGenerator, self).__init__()
        self.device = opt.device
        self.dim_latent_scene_noise = opt.dim_latent_scene_noise
        self.dim_latent_scene = opt.dim_latent_scene
        self.dim_latent_map = opt.dim_latent_map
        self.map_enc = MapEncoder(opt)
        self.scene_embedder_out = MLP(d_in=self.dim_latent_scene_noise + self.dim_latent_map,
                                      d_out=self.dim_latent_scene,
                                      d_hid=self.dim_latent_scene,
                                      n_layers=3,
                                      device=self.device)
        if opt.agents_decoder_model == 'GRU':
            self.agents_dec = AgentsDecoderGRU(opt, self.device)
        elif opt.agents_decoder_model == 'MLP':
            self.agents_dec = AgentsDecoderMLP(opt, self.device)
        else:
            raise ValueError
        # Debug - print parameter names:  [x[0] for x in self.named_parameters()]
        self.batch_size = opt.batch_size
        if self.batch_size != 1:
            raise NotImplementedError

    def forward(self, conditioning):
        """Standard forward"""
        map_feat = conditioning['map_feat']
        n_agents = conditioning['n_agents']
        map_latent = self.map_enc(map_feat)
        latent_noise = torch.randn(self.dim_latent_scene_noise, device=self.device)
        scene_latent = torch.concat([map_latent, latent_noise], dim=0)
        scene_latent = self.scene_embedder_out(scene_latent)
        agents_feat_vec_list = self.agents_dec(scene_latent, n_agents)
        agents_feat_vecs = torch.stack(agents_feat_vec_list)
        return agents_feat_vecs


#########################################################################################


class SceneDiscriminator(nn.Module):

    def __init__(self, opt):
        super(SceneDiscriminator, self).__init__()
        self.device = opt.device
        self.agent_feat_vec_coord_labels = opt.agent_feat_vec_coord_labels
        self.dim_agent_feat_vec = len(opt.agent_feat_vec_coord_labels)
        self.dim_latent_all_agents = opt.dim_latent_all_agents
        self.dim_latent_map = opt.dim_latent_map
        self.dim_latent_scene = opt.dim_latent_scene
        self.map_enc = MapEncoder(opt)
        self.agents_enc = PointNet(d_in=self.dim_agent_feat_vec,
                                   d_out=self.dim_latent_all_agents,
                                   d_hid=self.dim_latent_all_agents,
                                   n_layers=3,
                                   device=self.device)
        self.out_mlp = MLP(d_in=self.dim_latent_map + self.dim_latent_all_agents,
                           d_out=1,
                           d_hid=self.dim_latent_scene,
                           n_layers=3,
                           device=self.device)

    def forward(self, conditioning, agents_feat_vecs):
        """Standard forward."""
        map_feat = conditioning['map_feat']
        map_latent = self.map_enc(map_feat)
        agents_latent = self.agents_enc(agents_feat_vecs)
        scene_latent = torch.cat([map_latent, agents_latent])
        pred_fake = self.out_mlp(scene_latent)
        ''' 
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        '''
        return pred_fake
#########################################################################################
