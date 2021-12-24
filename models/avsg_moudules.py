import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers, opt, bias=True):
        super(MLP, self).__init__()
        assert n_layers >= 1
        self.device = opt.device
        self.use_layer_norm = opt.use_layer_norm
        self.d_in = d_in
        self.d_out = d_out
        self.d_hid = d_hid
        self.n_layers = n_layers
        layer_dims = [d_in] + (n_layers - 1) * [d_hid] + [d_out]
        modules_list = []
        for i_layer in range(n_layers - 1):
            layer_d_in = layer_dims[i_layer]
            layer_d_out = layer_dims[i_layer + 1]
            modules_list.append(nn.Linear(layer_d_in, layer_d_out, bias=bias, device=self.device))
            if self.use_layer_norm:
                modules_list.append(nn.LayerNorm(layer_d_out, device=self.device))
            modules_list.append(nn.ReLU())
        modules_list.append(nn.Linear(layer_dims[-2], d_out, bias=bias, device=self.device))
        self.net = nn.Sequential(*modules_list)
        self.layer_dims = layer_dims

    def forward(self, in_vec):
        return self.net(in_vec)


#########################################################################################


class PointNet(nn.Module):

    def __init__(self, d_in, d_out, d_hid, n_layers, opt):
        super(PointNet, self).__init__()
        self.device = opt.device
        self.use_layer_norm = opt.use_layer_norm
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
            self.matA[i_layer] = nn.Parameter(torch.zeros(layer_dims, device=self.device, requires_grad=True))
            self.matB[i_layer] = nn.Parameter(torch.zeros(layer_dims, device=self.device, requires_grad=True))
            self.register_parameter(name=f'matA_{i_layer}', param=self.matA[i_layer])
            self.register_parameter(name=f'matB_{i_layer}', param=self.matB[i_layer])
            # PyTorch's default initialization:
            nn.init.kaiming_uniform_(self.matA[i_layer], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.matB[i_layer], a=math.sqrt(5))
        self.out_layer = nn.Linear(d_hid, d_out, device=self.device)
        if self.use_layer_norm:
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
            if self.use_layer_norm:
                h = self.layer_normalizer(h)
            h = F.relu(h)
        # apply permutation invariant aggregation over all elements
        # max-pooling in our case
        h = torch.max(h, dim=0).values
        h = self.out_layer(h)
        return h

