import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

########################################################
# Networks that take patch info as well
########################################################

class Unet3dPatchGenerator(nn.Module):
    '''
    Generator for 3D Unet (adapted from Li et al. Context Matters: Graph-based Self-supervised Representation Learning for Medical Images and modified a bit)
    '''

    def __init__(self, input_nc, output_nc, norm_layer, add_final=False, patchfloat=False):
        super().__init__()

        self.add_final = add_final
        self.patchfloat = patchfloat
        # If continuous values of patch ids (x, y, z)
        if patchfloat:
            embed = []
            embed.append(nn.Linear(3, 64,))
            embed.append(nn.LeakyReLU())
            embed.append(nn.Linear(64, 64,))
            embed.append(nn.LeakyReLU())
            embed.append(nn.Linear(64, 32,))
            embed.append(nn.LeakyReLU())
            embed.append(nn.Linear(32, 32,))
            self.embed = nn.Sequential(*embed)
        else:
            self.embed = nn.Embedding(581, 32)

        # Encoder modules
        enc_modules = []
        enc_modules.append(BasicBlock3d(input_nc, 8, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(8, 8, 3, 2, norm_layer))  # 16
        enc_modules.append(BasicBlock3d(8, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 2, norm_layer)) # 8
        enc_modules.append(BasicBlock3d(16, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, norm_layer)) # 4
        self.encoder = nn.Sequential(*enc_modules)


        # Decoder modules
        dec_modules = []
        dec_modules.append(BasicBlock3d(64, 32, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(32, 16, 3, 1, norm_layer, upsample=2))
        dec_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(16, 8, 3, 1, norm_layer, upsample=2))
        dec_modules.append(BasicBlock3d(8, 8, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(8, 8, 3, 1, norm_layer))
        dec_modules.append(BasicBlock3d(8, 8, 3, 1, norm_layer, upsample=2))
        dec_modules.append(BasicBlock3d(8, output_nc, 3, 1, norm_layer=None, activ=None))
        self.decoder = nn.Sequential(*dec_modules)


    def forward(self, x, label):
        enc = self.encoder(x)                           # [B, 32, H, W, D]
        if self.patchfloat:
            embed = self.embed(label)[..., None, None, None]   # [B, 32, 1, 1, 1]
        else:
            embed = self.embed(label.long())[..., None, None, None]      # [B, 32, 1, 1, 1]
        #embed = embed.repeat(1, 1, enc.shape[2], enc.shape[3])
        embed = embed.repeat(1, 1, enc.shape[2], enc.shape[3], enc.shape[4])   # [B, 32, H, W]
        enc = torch.cat([enc, embed], 1)            # [B, 64, H, W]
        dec = self.decoder(enc)
        if self.add_final:
            dec = dec + x
        return dec


class Unet3dPatchDiscriminator(nn.Module):
    '''
    Discriminator for 3D Unet (adapted from Li et al. Context Matters: Graph-based Self-supervised Representation Learning for Medical Images and modified a bit)
    '''

    def __init__(self, input_nc, norm_layer, output_nc=1, patchfloat=False):
        super().__init__()
        self.patchfloat = patchfloat
        # Encoder modules
        enc_modules = []
        enc_modules.append(BasicBlock3d(input_nc, 8, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(8, 8, 3, 2, norm_layer))   # 16
        enc_modules.append(BasicBlock3d(8, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 2, norm_layer))  # 8
        enc_modules.append(BasicBlock3d(16, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, norm_layer))  # 4
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, norm_layer))  # 2
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, None))  # 1 channel, 1 spatial resolution
        self.encoder = nn.Sequential(*enc_modules)

        # Get fc
        fc_modules = []
        fc_modules.append(BasicBlock3d(64, 32, 1, 1, None))
        fc_modules.append(BasicBlock3d(32, 32, 1, 1, None))
        fc_modules.append(BasicBlock3d(32, output_nc, 1, 1, None, activ=None))
        self.fc_modules = nn.Sequential(*fc_modules)

        if patchfloat:
            embed = []
            embed.append(nn.Linear(3, 64,))
            embed.append(nn.LeakyReLU())
            embed.append(nn.Linear(64, 64,))
            embed.append(nn.LeakyReLU())
            embed.append(nn.Linear(64, 32,))
            embed.append(nn.LeakyReLU())
            embed.append(nn.Linear(32, 32,))
            self.embed = nn.Sequential(*embed)
        else:
            self.embed = nn.Embedding(581, 32)



    def forward(self, X, lab):
        x = self.encoder(X)  # [B, 32, H, W, D]
        if self.patchfloat:
            embed = self.embed(lab)[..., None, None, None]
        else:
            embed = self.embed(lab.long())[..., None, None, None]
        #print(x.shape, embed.shape)
        #input()
        #print(x.shape, embed.shape, X.shape, lab.shape)
        embed = embed.repeat(1, 1, x.shape[2], x.shape[3], x.shape[4])   # [B, 32, H, W]
        x = torch.cat([x, embed], 1)
        x = self.fc_modules(x)
        return x



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


class Unet3dGenerator(nn.Module):
    '''
    Generator for 3D Unet (adapted from Li et al. Context Matters: Graph-based Self-supervised Representation Learning for Medical Images and modified a bit)
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


class Unet3dDiscriminator(nn.Module):
    '''
    Discriminator for 3D Unet (adapted from Li et al. Context Matters: Graph-based Self-supervised Representation Learning for Medical Images and modified a bit)
    '''

    def __init__(self, input_nc, norm_layer, output_nc=1):
        super().__init__()
        # Encoder modules
        enc_modules = []
        enc_modules.append(BasicBlock3d(input_nc, 8, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(8, 8, 3, 2, norm_layer))   # 16
        enc_modules.append(BasicBlock3d(8, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(16, 16, 3, 2, norm_layer))  # 8
        enc_modules.append(BasicBlock3d(16, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, norm_layer))  # 4
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, 32, 3, 2, norm_layer))  # 2
        enc_modules.append(BasicBlock3d(32, 32, 3, 1, norm_layer))
        enc_modules.append(BasicBlock3d(32, output_nc, 3, 2, None))  # 1 channel, 1 spatial resolution
        self.encoder = nn.Sequential(*enc_modules)

    def forward(self, x):
        return self.encoder(x)


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
    model = Unet3dPatchDiscriminator(1, nn.BatchNorm3d)
    #model = Unet3dGenerator(1, 1, nn.BatchNorm3d)
    inp = torch.rand(5, 1, 32, 32, 32)
    lab = torch.randint(581, size=(5, ))
    out = model(inp, lab)
    print(inp.shape, out.shape)
