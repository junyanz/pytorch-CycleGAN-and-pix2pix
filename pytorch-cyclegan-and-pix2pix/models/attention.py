import torch.nn as nn
import torch



class AttentionBlock(nn.Module):

    def __init__(self, in_channels=1, features=64):
        super(AttentionBlock, self).__init__()

        self.downsampling_branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True)
        )

        self.downsampling_branch_2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True)
        )

        self.downsampling_branch_3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True)
        )

        def residual_block(channels):
            return nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(channels)
            )

        self.residual_blocks = nn.ModuleList([
            residual_block(features * 4) for _ in range(9)
        ])


        self.upsampling_weight_branch = nn.Sequential(
            
            nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features * 2, features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features, features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1)
        )

        self.upsampling_content_branch = nn.Sequential(
            
            nn.ConvTranspose2d(features * 4,features * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features * 2,features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(features, features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),

            nn.Conv2d(features, in_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, x):

        original_x = x

        x = self.downsampling_branch_1(x)
        x = self.downsampling_branch_2(x)
        x = self.downsampling_branch_3(x)

        for residual_block in self.residual_blocks:
            x = torch.relu(x + residual_block(x))
            
        I_content = self.upsampling_content_branch(x)

        weights = self.upsampling_weight_branch(x)
        w_content = weights[:, 0:1, :, :]  # First channel
        w_input = weights[:, 1:2, :, :]     # Second channel
        
        I_att = I_content * w_content + original_x * w_input

        return I_att, I_content

