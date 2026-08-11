"""3D convolutional LSTM, used as an alternative temporal module (discriminator_temporal_mode='convlstm3d')
for the spatiotemporal discriminator in discriminators_temporal.py.

All gates use Conv3d (not Conv2d): the recurrence runs over the T axis, but each per-timestep update
is itself a 3D convolution over the (d, h, w) spatial bottleneck, so spatial structure is preserved
at every step rather than being flattened away.
"""

import torch
import torch.nn as nn


class ConvLSTM3DCell(nn.Module):
    """One ConvLSTM3D layer's recurrent cell.

    Input: x_t of shape (N, input_channels, d, h, w), plus the previous (h, c) state, each
    (N, hidden_channels, d, h, w). All four gates (input, forget, output, cell) are produced by a
    single Conv3d over the channel-concatenation of x_t and h_{t-1}, following the standard ConvLSTM
    formulation (Shi et al., 2015) generalized from Conv2d to Conv3d.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # odd kernel_size -> same spatial size in/out
        self.conv = nn.Conv3d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev):
        # x: (N, input_channels, d, h, w); h_prev, c_prev: (N, hidden_channels, d, h, w)
        combined = torch.cat([x, h_prev], dim=1)  # (N, input_channels + hidden_channels, d, h, w)
        gates = self.conv(combined)  # (N, 4*hidden_channels, d, h, w)
        i, f, o, g = torch.chunk(gates, 4, dim=1)  # each (N, hidden_channels, d, h, w)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size, spatial_size, device, dtype):
        d, h, w = spatial_size
        shape = (batch_size, self.hidden_channels, d, h, w)
        return torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)


class ConvLSTM3D(nn.Module):
    """Stack of ConvLSTM3DCell layers, run over the T axis of a (B, T, F, d, h, w) input.

    Returns ALL hidden states of the last layer, i.e. (B, T, hidden_channels, d, h, w) -- not just the
    final timestep -- since every T position needs its own temporally-conditioned features (the
    discriminator that consumes this must supervise every timestep, not just the last).
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        cells = []
        for layer in range(num_layers):
            in_ch = input_channels if layer == 0 else hidden_channels
            cells.append(ConvLSTM3DCell(in_ch, hidden_channels, kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        # x: (B, T, F, d, h, w) -> (B, T, hidden_channels, d, h, w)
        assert x.dim() == 6, f"ConvLSTM3D expected (B, T, F, d, h, w), got {tuple(x.shape)}"
        B, T, F, d, h, w = x.shape
        layer_input = x
        for cell in self.cells:
            h_t, c_t = cell.init_hidden(B, (d, h, w), device=x.device, dtype=x.dtype)
            outputs = []
            for t in range(T):
                h_t, c_t = cell(layer_input[:, t], h_t, c_t)  # (B, hidden_channels, d, h, w)
                outputs.append(h_t)
            layer_input = torch.stack(outputs, dim=1)  # (B, T, hidden_channels, d, h, w) -- all hidden states
        return layer_input
