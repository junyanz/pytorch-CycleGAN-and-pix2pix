"""Spatiotemporal PatchGAN discriminator variants for fMRI-sequence CycleGAN training.

TemporalNLayerDiscriminator wraps the existing NLayerDiscriminator architecture (models3D/networks.py)
with a temporal module (TCN or ConvLSTM3D) inserted at a shallow spatial bottleneck, for
discriminator_temporal_mode in ("tcn", "convlstm3d"). It deliberately does NOT touch or reuse
NLayerDiscriminator's own nn.Module structure/state_dict -- it rebuilds an equivalent stem+body conv
stack from scratch with the exact same channel/stride progression, so that the original
NLayerDiscriminator (used when discriminator_temporal_mode == "none") is completely unaffected: same
architecture, same forward, same state_dict keys, same checkpoint compatibility.

Axis convention: (B, C, T, H, W, D) -- C before T -- matching ResnetGenerator's use_temporal_module
convention exactly (see models3D/temporal.py), so a temporal generator's output can be fed directly
into a temporal discriminator with no reconciling permute needed at the handoff.
"""

import functools

import torch.nn as nn

from .convlstm3d import ConvLSTM3D
from .temporal import TemporalTCN, bottleneck_to_temporal_sequences, temporal_sequences_to_bottleneck


class TemporalNLayerDiscriminator(nn.Module):
    """PatchGAN discriminator with a temporal module (TCN or ConvLSTM3D) inserted after a shallow
    spatial stem and before the remaining spatial downsampling ("body") layers.

    Input:  (B, C, T, H, W, D)
    Output: (B, 1, T, d_patch, h_patch, w_patch) -- one patch-map per input timestep, every timestep
            directly supervised (T is never touched by any conv layer; no temporal downsampling).

    Architecture: identical conv/stride/channel progression to NLayerDiscriminator(ndf, n_layers), just
    split into two pieces at `stem_downsample_layers` so a temporal module can sit between them:
        stem  = the first `stem_downsample_layers` stride-2 downsampling stages (limited early
                 downsampling: enough to make (d,h,w) tractable for per-location temporal processing,
                 without collapsing all spatial detail before the temporal module sees it).
        body  = the remaining stride-2 stage(s), the stride-1 stage, and the final 1-channel patch conv
                 -- i.e. exactly NLayerDiscriminator's own tail, unchanged in spirit.
    """

    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm3d,
        temporal_mode="tcn",
        stem_downsample_layers=2,
        temporal_dilations=(1, 2, 4),
        temporal_dropout=0.1,
        temporal_norm_groups=8,
        convlstm_hidden_channels=None,
        convlstm_num_layers=1,
        convlstm_kernel_size=3,
    ):
        super().__init__()
        if temporal_mode not in ("tcn", "convlstm3d"):
            raise NotImplementedError("temporal_mode [%s] is not implemented" % temporal_mode)
        if not (1 <= stem_downsample_layers < n_layers):
            raise ValueError("stem_downsample_layers must be in [1, n_layers-1] so the body keeps at least one downsampling stage (got stem_downsample_layers=%d, n_layers=%d)" % (stem_downsample_layers, n_layers))

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw, padw = 4, 1

        # --- stem: first stem_downsample_layers stride-2 stages, exactly matching NLayerDiscriminator's
        # own channel/stride progression (see models3D/networks.py:NLayerDiscriminator) ---
        stem = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult, nf_mult_prev = 1, 1
        for n in range(1, stem_downsample_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            stem += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        self.stem = nn.Sequential(*stem)
        stem_channels = ndf * nf_mult  # F: derived from the actual stem, never hardcoded

        # --- body: remaining stride-2 stage(s), the stride-1 stage, and the final 1-channel patch conv
        # -- continues from stem_downsample_layers so the overall progression (and receptive field) is
        # identical to running the full, unmodified NLayerDiscriminator ---
        body = []
        for n in range(stem_downsample_layers, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            body += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        body += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        body += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.body = nn.Sequential(*body)

        self.temporal_mode = temporal_mode
        self.stem_channels = stem_channels
        if temporal_mode == "tcn":
            self.temporal_module = TemporalTCN(
                feature_channels=stem_channels,
                dilations=temporal_dilations,
                dropout=temporal_dropout,
                norm_groups=temporal_norm_groups,
            )
            self.temporal_adapter = None
        else:  # convlstm3d
            hidden_channels = convlstm_hidden_channels or stem_channels
            self.temporal_module = ConvLSTM3D(
                input_channels=stem_channels,
                hidden_channels=hidden_channels,
                kernel_size=convlstm_kernel_size,
                num_layers=convlstm_num_layers,
            )
            # If ConvLSTM3D's hidden size differs from what the body expects, adapt with a learnable 1x1x1 conv.
            self.temporal_adapter = nn.Conv3d(hidden_channels, stem_channels, kernel_size=1) if hidden_channels != stem_channels else None

    def forward(self, input):
        """input: (B, C, T, H, W, D) -> output: (B, 1, T, d_patch, h_patch, w_patch)"""
        assert input.dim() == 6, f"TemporalNLayerDiscriminator expects a 6D (B,C,T,H,W,D) input, got shape {tuple(input.shape)}"
        B, C, T, H, W, D = input.shape

        # Merge B and T so the shared spatial stem (Conv3d, which only understands (N,C,D,H,W)) can
        # process every time point independently. Identical merge step to ResnetGenerator's
        # use_temporal_module (same input convention, same permute indices).
        x = input.permute(0, 2, 1, 5, 3, 4).contiguous()  # (B, T, C, D, H, W)
        x = x.reshape(B * T, C, D, H, W)  # (B*T, C, D, H, W) -- Conv3d input convention

        z = self.stem(x)  # (B*T, F, d, h, w) -- shallow spatial bottleneck
        _, F, d, h, w = z.shape
        assert F == self.stem_channels, f"stem channel mismatch: stem produced F={F}, temporal module configured for {self.stem_channels}"

        if self.temporal_mode == "tcn":
            z_seq, _ = bottleneck_to_temporal_sequences(z, B, T)  # (B*d*h*w, F, T)
            z_seq = self.temporal_module(z_seq)  # (B*d*h*w, F, T), shape unchanged
            z = temporal_sequences_to_bottleneck(z_seq, B, T, d, h, w)  # (B*T, F, d, h, w)
        else:  # convlstm3d
            z_bt = z.reshape(B, T, F, d, h, w)  # (B, T, F, d, h, w)
            z_bt = self.temporal_module(z_bt)  # (B, T, F_hidden, d, h, w) -- ALL hidden states, not just the last
            Bh, Th, Fh, dh, hh, wh = z_bt.shape
            z = z_bt.reshape(Bh * Th, Fh, dh, hh, wh)  # (B*T, F_hidden, d, h, w)
            if self.temporal_adapter is not None:
                z = self.temporal_adapter(z)  # (B*T, F, d, h, w) -- back to stem_channels for the body

        patch = self.body(z)  # (B*T, 1, d_patch, h_patch, w_patch)
        _, _, d_p, h_p, w_p = patch.shape
        patch = patch.reshape(B, T, 1, d_p, h_p, w_p)  # (B, T, 1, d_patch, h_patch, w_patch) -- every T supervised
        patch = patch.permute(0, 2, 1, 3, 4, 5).contiguous()  # (B, 1, T, d_patch, h_patch, w_patch) -- C before T, matches generator
        return patch
