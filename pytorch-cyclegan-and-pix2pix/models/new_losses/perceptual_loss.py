import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch




class PerceptualLoss(nn.Module):

    def __init__(self, device):

        super(PerceptualLoss, self).__init__()

        # Import the pre-trained ResNet34 Model
        resNet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.resNet34conv1 = resNet34.conv1
        self.resNet34bn1 = resNet34.bn1
        self.resNet34relu = resNet34.relu
        self.resNet34maxpool = resNet34.maxpool

        # Get only feature extraction layers
        self.resNet34Layer1 = resNet34.layer1
        self.resNet34Layer2 = resNet34.layer2
        self.resNet34Layer3 = resNet34.layer3
        self.resNet34Layer4 = resNet34.layer4

        # Fix the parameters in ResNet34
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

        # Put on GPU if possible
        self.to(device)

    def _prepare_input(self, x):
        # Convert MRI to 3-Channel RGB for ResNet34
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x


    def extract_features(self, x):

        # Prepare input
        x = self._prepare_input(x)

        # Extract features by passing through ResNet34

        features = []

        x = self.resNet34conv1(x)
        x = self.resNet34bn1(x)
        x = self.resNet34relu(x)
        x = self.resNet34maxpool(x)

        x = self.resNet34Layer1(x)
        features.append(x)
        
        x = self.resNet34Layer2(x)
        features.append(x)
        
        x = self.resNet34Layer3(x)
        features.append(x)
        
        x = self.resNet34Layer4(x)
        features.append(x)
        
        return features
    

    def gram_matrix(self, features):

        B, C, H, W = features.shape
        
        # Re-shape features
        features_reshaped = features.view(B, C, H * W)
        
        # Calculate gram matrix
        gram_matrix = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2))
        
        # Normalise
        gram_matrix = gram_matrix / (C * H * W)
        
        return gram_matrix


    def loss_per_C(self, fake_B, real_B, fake_A, real_A):

        fake_B_features = self.extract_features(fake_B)
        real_A_features  = self.extract_features(real_A)

        fake_A_features  = self.extract_features(fake_A)
        real_B_features = self.extract_features(real_B)
            
        return (
            F.mse_loss(fake_B_features[3], real_A_features[3])
            + F.mse_loss(fake_A_features[3], real_B_features[3])
        )


    def loss_per_S(self, fake_B, real_B, fake_A, real_A):
          
        fake_B_features = self.extract_features(fake_B)
        real_B_features = self.extract_features(real_B)
        fake_A_features = self.extract_features(fake_A)
        real_A_features = self.extract_features(real_A)
        
        loss = 0

        for i in range(4):

            loss += F.mse_loss(
                self.gram_matrix(fake_B_features[i]),
                self.gram_matrix(real_B_features[i])
            )

            loss += F.mse_loss(
                self.gram_matrix(fake_A_features[i]),
                self.gram_matrix(real_A_features[i])
            )

        return loss


