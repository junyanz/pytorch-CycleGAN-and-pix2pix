import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import collections
from itertools import repeat
from typing import List, Union, Tuple
#from custom_layers import ReflectionPad3d

# For now, we are just using zero padding
class ReflectionPad3d:
    def __init__(self,):
        raise NotImplementedError

class ReplicationPad3d:
    def __init__(self,):
        raise NotImplementedError


class ResnetGenerator3d(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm3d, use_dropout=False, n_blocks=6, padding_type='zero',
            add_final=True, patchfloat=True):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        self.add_final = add_final
        self.patchfloat = patchfloat

        assert(n_blocks >= 0)
        super(ResnetGenerator3d, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        encoder = [ #ReflectionPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            encoder += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        ## Add decoder here
        decoder = []
        mult = 2 ** n_downsampling
        embed_size = mult * ngf
        if patchfloat:
            embed = [nn.Linear(3, embed_size)]
            for _ in range(3):
                embed.append(nn.ELU())
                embed.append(nn.Linear(embed_size, embed_size))
            self.embed = nn.Sequential(*embed)
        else:
            self.embed = nn.Embedding(581, embed_size)

        # A small conv3d to merge information from encoder and patch info
        self.converter = nn.Conv3d(ngf * mult * 2, ngf * mult, kernel_size=3, stride=1, padding=1, bias=use_bias)

        for i in range(n_blocks):       # add ResNet blocks
            decoder += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]


        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # Replace ConvTranspose3d with Conv3d followed by upsampling
            '''
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            '''
            decoder += [nn.Conv3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      nn.Upsample(scale_factor=2, mode='trilinear'),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]


        #model += [ReflectionPad3d(3)]
        #model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        decoder += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=3)]
        decoder += [nn.Tanh()]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)


    def forward(self, x, Patchloc):
        """Standard forward"""
        # Accomodate if there is partitions information, just ignore it
        if isinstance(Patchloc, (list, tuple)):
            patchloc = Patchloc[0]
        else:
            patchloc = Patchloc

        # return self.model(input)
        enc = self.encoder(x)
        B, C, H, W, D = enc.shape
        # Get patch location
        embed = self.embed(patchloc)[..., None, None, None]
        embed = embed.repeat(1, 1, H, W, D)
        # Concatenate embedding and encoded information
        enc = torch.cat([enc, embed], 1)
        enc = self.converter(enc)  # merge information
        dec = self.decoder(enc)
        if self.add_final:
            dec = dec + x
        return dec



class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


#############################
## PatchGAN discriminator
#############################

class NLayerDiscriminator3d(nn.Module):
    """Defines a PatchGAN discriminator for 3D which takes patch information"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm3d, patchfloat=True, partitions=None):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3d, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        # Rest of the sequence
        sequence_final = []
        embed_size = nf_mult * ndf

        nf_mult_prev = nf_mult*2   # also considering embed size
        nf_mult = min(2 ** n_layers, 8)
        sequence_final += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence_final += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*sequence_final)
        # Also load embedding for patches
        self.patchfloat = patchfloat
        if not patchfloat:   # int are provided
            self.embed = nn.Embedding(581, embed_size)
        else:
            embed = []
            embed.append(nn.Linear(3, embed_size))
            for i in range(3):
                embed.append(nn.ELU())
                embed.append(nn.Linear(embed_size, embed_size))
            self.embed = nn.Sequential(*embed)

        # Load partitions if exists
        self.partitions = partitions
        if partitions is not None:
            self.partembed = nn.Embedding(partitions, embed_size)

        pass


    def forward(self, input, Patchloc):
        """Standard forward."""
        if self.partitions is not None:    # There exist some partitions
            patchloc, partloc = Patchloc
        else:
            patchloc, partloc = Patchloc, None

        enc = self.model(input)   # [B, C, H, W, D]
        B, C, H, W, D = enc.shape
        # If int indices are given, then just use the long tensors for embedding
        if not self.patchfloat:
            patchloc = patchloc.long()

        embed = self.embed(patchloc)[..., None, None, None]  # [B, E, 1, 1, 1]
        # If part location is not none, just add it
        if partloc is not None:
            embed = embed + self.partembed(partloc)[..., None, None, None]  # [B, E, 1, 1, 1]

        embed = embed.repeat(1, 1, H, W, D)
        # Concatenate channels along channel
        enc = torch.cat([enc, embed], 1)
        dec = self.fc(enc)
        return dec


if __name__ == "__main__":
    testnet = 0
    if testnet == 0:
        net = ResnetGenerator3d(1, 1, n_blocks=9)
        print(net.__class__.__name__)
        a = torch.rand(5, 1, 32, 32, 32)
        enc = torch.randn(5, 3)
        part = torch.randint(5, size=(5,))
        out = net(a, (enc, part))
        print(out.shape)
    elif testnet == 1:
        net = NLayerDiscriminator3d(1, partitions=5)
        a = torch.rand(5, 1, 32, 32, 32)
        enc = torch.randn(5, 3)
        part = torch.randint(5, size=(5,))
        out = net(a, (enc, part))
        print(out.shape)

