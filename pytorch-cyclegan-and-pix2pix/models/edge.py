import torch.nn as nn
import torch




class EdgeBlock(nn.Module):

    def __init__(self, in_channels=1, features=32):
        super().__init__()

        # Define the edge U-Net

        # Encoder

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder

        self.up1 = nn.ConvTranspose2d(features * 8, features * 4, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(features * 2, features, 3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.Sequential(
            nn.Conv2d(features * 2, features, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.dec4 = nn.Sequential(
            nn.Conv2d(features, in_channels, 1),
            # nn.ReLU(inplace=True) Stupid incorrect diagram AGH
        )

    def forward(self, x):

        # Run the encoder side
        enc1_output = self.enc1(x)
        pool1_output = self.pool1(enc1_output)

        enc2_output = self.enc2(pool1_output)
        pool2_output = self.pool2(enc2_output)

        enc3_output = self.enc3(pool2_output)
        pool3_output = self.pool3(enc3_output)

        enc4_output = self.enc4(pool3_output)

        # Run the decoder side
        up1_output = self.up1(enc4_output)
        concentration1 = torch.cat([ up1_output, enc3_output ], dim=1)
        dec1_output = self.dec1(concentration1)
        
        up2_output = self.up2(dec1_output)
        concentration2 = torch.cat([ up2_output, enc2_output ], dim=1)
        dec2_output = self.dec2(concentration2)

        up3_output = self.up3(dec2_output)
        concentration3 = torch.cat([ up3_output, enc1_output ], dim=1)
        dec3_output = self.dec3(concentration3)

        dec4_output = self.dec4(dec3_output)

        return dec4_output



