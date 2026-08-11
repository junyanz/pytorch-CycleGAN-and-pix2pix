"""Temporal convolutional module for the 3D CycleGAN generator bottleneck.

Applies a dilated, non-causal 1D temporal residual network independently at every
(subject, bottleneck spatial location) pair, operating along the fMRI time axis T.
This is separate from the spatial encoder/decoder (Conv3d) in networks.py: it only
ever sees (N, F, T) tensors and never mixes across the N (effective batch) dimension.

Receptive field: each TemporalResidualBlock has two kernel_size=3 Conv1d layers at
the same dilation d, so it extends the receptive field by 2 * (kernel_size - 1) * d
= 4*d time points. Summed over dilations (1, 2, 4):
    R = 1 + 4 * (1 + 2 + 4) = 29 time points
which comfortably covers a T=20 fMRI chunk (see TemporalTCN docstring).
"""

import torch
import torch.nn as nn


def _largest_valid_num_groups(num_channels, requested_groups):
    """Largest divisor of num_channels that does not exceed requested_groups."""
    requested_groups = max(1, min(requested_groups, num_channels))
    for g in range(requested_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # unreachable: g=1 always divides num_channels


class TemporalResidualBlock(nn.Module):
    """One dilated, non-causal residual block over (N, F, T) sequences.

        Input
          |---------------------------- identity ----------------------------|
          v                                                                   |
        Conv1d -> GroupNorm -> GELU -> Dropout -> Conv1d -> GroupNorm -> GELU -> Dropout -> (+) -> out

    Non-causal (symmetric padding, no future masking) because this is offline fMRI
    correction: both earlier and later time points in the chunk are available.
    """

    def __init__(self, feature_channels, dilation, dropout=0.1, norm_groups=8):
        super().__init__()
        num_groups = _largest_valid_num_groups(feature_channels, norm_groups)
        padding = dilation  # kernel_size=3, dilation=d -> padding=d preserves (N,F,T) length

        self.conv1 = nn.Conv1d(feature_channels, feature_channels, kernel_size=3, stride=1, dilation=dilation, padding=padding)
        self.norm1 = nn.GroupNorm(num_groups, feature_channels)
        self.conv2 = nn.Conv1d(feature_channels, feature_channels, kernel_size=3, stride=1, dilation=dilation, padding=padding)
        self.norm2 = nn.GroupNorm(num_groups, feature_channels)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (N, F, T) -> (N, F, T)
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        out = self.dropout2(out)
        return residual + out


class TemporalTCN(nn.Module):
    """Stack of dilated non-causal TemporalResidualBlocks over (N, F, T) sequences.

    N is an *effective* batch: the caller is expected to have already merged
    (subject, bottleneck spatial location) into N, so this module applies the exact
    same temporal network independently and identically at every spatial location,
    for every subject, without ever mixing across them.

    Receptive field with kernel_size=3 (2 convs/block) and dilations (1, 2, 4):
        R = 1 + 4 * sum(dilations) = 1 + 4*(1+2+4) = 29 time points.
    """

    def __init__(self, feature_channels, dilations=(1, 2, 4), dropout=0.1, norm_groups=8):
        super().__init__()
        self.feature_channels = feature_channels
        self.blocks = nn.ModuleList([TemporalResidualBlock(feature_channels, dilation=d, dropout=dropout, norm_groups=norm_groups) for d in dilations])

    def forward(self, x):
        # x: (N, F, T) -> (N, F, T)
        assert x.dim() == 3 and x.shape[1] == self.feature_channels, f"TemporalTCN expected (N, {self.feature_channels}, T), got {tuple(x.shape)}"
        for block in self.blocks:
            x = block(x)
        return x


def bottleneck_to_temporal_sequences(z, B, T):
    """(B*T, F, d, h, w) -> (B*d*h*w, F, T): one temporal sequence per (subject, bottleneck location).

    Returns (z_seq, (F, d, h, w)) so the caller can invert with temporal_sequences_to_bottleneck.
    Factored out (rather than inlined in the generator) so it can be tested standalone for an
    exact, value-preserving round trip when the temporal module itself is bypassed.
    """
    _, F, d, h, w = z.shape
    z = z.reshape(B, T, F, d, h, w)  # (B, T, F, d, h, w)
    z = z.permute(0, 3, 4, 5, 2, 1).contiguous()  # (B, d, h, w, F, T)
    z = z.reshape(B * d * h * w, F, T)  # (B*d*h*w, F, T) -- effective batch, features, time
    return z, (F, d, h, w)


def temporal_sequences_to_bottleneck(z_seq, B, T, d, h, w):
    """(B*d*h*w, F, T) -> (B*T, F, d, h, w): exact inverse of bottleneck_to_temporal_sequences."""
    F = z_seq.shape[1]
    z = z_seq.reshape(B, d, h, w, F, T)  # (B, d, h, w, F, T)
    z = z.permute(0, 5, 4, 1, 2, 3).contiguous()  # (B, T, F, d, h, w)
    z = z.reshape(B * T, F, d, h, w)  # (B*T, F, d, h, w) -- input to the spatial decoder
    return z
