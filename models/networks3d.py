import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class Unet3dGenerator(nn.Module):
    '''
    Generator for 3D Unet (adapted from Li et al. Context Matters: Graph-based Self-supervised Representation Learning for Medical Images)
    '''

    def __init__(self, input_nc, output_nc, norm_layer):
        super().__init__()

        # Encoder modules
        enc_modules = []
        enc_modules.append(BasicBlock3d(input_nc, 8, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(8, 8, 3, 2, norm_layer))
        enc_modules.append(BasicBlock3d(8, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 2, norm_layer))
        enc_modules.append(BasicBlock3d(16, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, norm_layer))
        self.encoder = nn.Sequential(*enc_modules)

        # Decoder modules
        dec_modules = []
        dec_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(32, 16, 3, 1, norm_layer, upsample=2))
        dec_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(16, 8, 3, 1, norm_layer, upsample=2))
        dec_modules.append(BasicBlock3d(8, 8, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(8, 8, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(8, 8, 3, 1, norm_layer, upsample=2))
        dec_modules.append(BasicBlock3d(8, output_nc, 3, 1, norm_layer=None, activ=None))
        self.decoder = nn.Sequential(*dec_modules)


    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec



class BasicBlock3d(nn.Module):
    '''
    Basic block for UNet3d (has support for downsampling and upsampling)
    '''
    def __init__(self, in_c, out_c, kernel_size, stride, norm_layer, activ=nn.ELU(), upsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            padding = int((kernel_size-1)/2)
        else:
            padding = [int((x-1)/2) for x in kernel_size]

        self.conv = nn.Conv3d(in_c, out_c, kernel_size, stride=stride, padding=padding)
        self.layer = norm_layer(out_c) if norm_layer is not None else lambda x: x
        self.activ = activ
        if upsample is None:
            self.upsample = lambda x: x
        else:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='trilinear')


    def forward(self, x):
        x = self.conv(x)
        x = self.layer(x)
        if self.activ is not None:
            x = self.activ(x)
        x = self.upsample(x)
        return x


if __name__ == "__main__":
    model = Unet3dGenerator(1, 1, nn.BatchNorm3d)
    inp = torch.rand(5, 1, 32, 32, 32)
    out = model(inp)
    print(inp.shape, out.shape)
