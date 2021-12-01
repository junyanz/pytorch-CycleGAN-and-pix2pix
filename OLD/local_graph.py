import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """A positional embedding module.
        Useful to inject the position of sequence elements in local graphs

        :param d_model: feature size
        :type d_model: int
        :param max_len:max length of the sequences, defaults to 5000
        :type max_len: int, optional
        """
        super().__init__()

        # Positional Encoder
        pe = torch.zeros(max_len, d_model)
        t = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        log_value = torch.log(torch.tensor([1e4])).item()
        omega = torch.exp((-log_value / d_model) * torch.arange(0, d_model, 2).float())
        pe[:, 0::2] = torch.sin(t * omega)
        pe[:, 1::2] = torch.cos(t * omega)
        self.register_buffer("static_embedding", pe.unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape batch_size x num_agents x sequence_length x d_model
        """
        return self.static_embedding[: x.shape[2], :]


class LocalMLP(nn.Module):
    def __init__(self, dim_in: int, use_norm: bool = True):
        """a Local 1 layer MLP

        :param dim_in: feat in size
        :type dim_in: int
        :param use_norm: if to apply layer norm, defaults to True
        :type use_norm: bool, optional
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_in, bias=not use_norm)
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(dim_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward of the module

        :param x: input tensor (..., dim_in)
        :type x: torch.Tensor
        :return: output tensor (..., dim_in)
        :rtype: torch.Tensor
        """
        x = self.linear(x)
        if hasattr(self, "norm"):
            x = self.norm(x)
        x = F.relu(x, inplace=True)
        return x


class LocalSubGraphLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        """Local subgraph layer

        :param dim_in: input feat size
        :type dim_in: int
        :param dim_out: output feat size
        :type dim_out: int
        """
        super(LocalSubGraphLayer, self).__init__()
        self.mlp = LocalMLP(dim_in)
        self.linear_remap = nn.Linear(dim_in * 2, dim_out)

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """Forward of the model

        :param x: input tensor
        :tensor (B,N,P,dim_in)
        :param invalid_mask: invalid mask for x
        :tensor invalid_mask (B,N,P)
        :return: output tensor (B,N,P,dim_out)
        :rtype: torch.Tensor
        """
        # x input -> polys * num_vectors * embedded_vector_length
        _, num_vectors, _ = x.shape
        # x mlp -> polys * num_vectors * dim_in
        x = self.mlp(x)
        # compute the masked max for each feature in the sequence

        masked_x = x.masked_fill(invalid_mask[..., None] > 0, float("-inf"))
        x_agg = masked_x.max(dim=1, keepdim=True).values
        # repeat it along the sequence length
        x_agg = x_agg.repeat(1, num_vectors, 1)
        x = torch.cat([x, x_agg], dim=-1)
        x = self.linear_remap(x)  # remap to a possibly different feature length
        return x


class LocalSubGraph(nn.Module):
    def __init__(self, num_layers: int, dim_in: int) -> None:
        """PointNet-like local subgraph - implemented as a collection of local graph layers

        :param num_layers: number of LocalSubGraphLayer
        :type num_layers: int
        :param dim_in: input, hidden, output dim for features
        :type dim_in: int
        """
        super(LocalSubGraph, self).__init__()
        assert num_layers > 0
        self.layers = nn.ModuleList()
        self.dim_in = dim_in
        for _ in range(num_layers):
            self.layers.append(LocalSubGraphLayer(dim_in, dim_in))

    def forward(self, x: torch.Tensor, invalid_mask: torch.Tensor, pos_enc: torch.Tensor) -> torch.Tensor:
        """Forward of the module:
        - Add positional encoding
        - Forward to layers
        - Aggregates using max
        (calculates a feature descriptor per element - reduces over points)

        :param x: input tensor (B,N,P,dim_in)
        :type x: torch.Tensor
        :param invalid_mask: invalid mask for x (B,N,P)
        :type invalid_mask: torch.Tensor
        :param pos_enc: positional_encoding for x
        :type pos_enc: torch.Tensor
        :return: output tensor (B,N,P,dim_in)
        :rtype: torch.Tensor
        """
        batch_size, polys_num, seq_len, vector_size = x.shape

        x += pos_enc
        # exclude completely invalid sequences from local subgraph to avoid NaN in weights
        x_flat = x.view(-1, seq_len, vector_size)
        invalid_mask_flat = invalid_mask.view(-1, seq_len)
        # (batch_size x (1 + M),)
        valid_polys = ~invalid_mask.all(-1).flatten()
        # valid_seq x seq_len x vector_size
        x_to_process = x_flat[valid_polys]
        mask_to_process = invalid_mask_flat[valid_polys]
        for layer in self.layers:
            x_to_process = layer(x_to_process, mask_to_process)

        # aggregate sequence features
        x_to_process = x_to_process.masked_fill(mask_to_process[..., None] > 0, float("-inf"))
        # valid_seq x vector_size
        x_to_process = torch.max(x_to_process, dim=1).values

        # restore back the batch
        x = torch.zeros_like(x_flat[:, 0])
        x[valid_polys] = x_to_process
        x = x.view(batch_size, polys_num, self.dim_in)
        return x
