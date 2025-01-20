import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import pathlib
import time
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib.request
import tarfile

# Set the GPU device (change this to the GPU you want to use)
GPU_ID = 1
if torch.cuda.is_available():
    device = torch.device(f'cuda:{GPU_ID}')
    print(f"Using GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")
else:
    device = torch.device('cpu')
    print("GPU not available, using CPU")

# Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100

def load(image_file):
    """Load and preprocess image file."""
    image = Image.open(image_file)
    image = transforms.ToTensor()(image)
    
    w = image.shape[2] // 2
    input_image = image[:, :, w:]
    real_image = image[:, :, :w]
    
    return input_image.float(), real_image.float()

def resize(input_image, real_image, height, width):
    """Resize images to the specified height and width."""
    input_image = transforms.Resize((height, width))(input_image)
    real_image = transforms.Resize((height, width))(real_image)
    return input_image, real_image

def random_crop(input_image, real_image):
    """Randomly crop both images to IMG_HEIGHT x IMG_WIDTH."""
    stacked = torch.cat([input_image, real_image], dim=0)
    cropped = transforms.RandomCrop((IMG_HEIGHT, IMG_WIDTH))(stacked)
    return cropped[:3], cropped[3:]

def normalize(input_image, real_image):
    """Normalize images to [-1, 1] range."""
    input_image = (input_image * 2) - 1
    real_image = (real_image * 2) - 1
    return input_image, real_image

def random_jitter(input_image, real_image):
    """Apply random jittering to images."""
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    
    if torch.rand(1) > 0.5:
        input_image = transforms.RandomHorizontalFlip(p=1.0)(input_image)
        real_image = transforms.RandomHorizontalFlip(p=1.0)(real_image)
    
    return input_image, real_image

class Pix2PixDataset(data.Dataset):
    """Enhanced dataset class with better error handling and image validation."""
    def __init__(self, path, is_train=True, load_size=286, crop_size=256):
        super().__init__()
        self.path = path
        self.is_train = is_train
        self.load_size = load_size
        self.crop_size = crop_size
        
        # Get image paths and validate directory
        self.image_paths = list(path.glob('*.jpg'))
        self.image_paths.extend(list(path.glob('*.png')))
        
        if not self.image_paths:
            raise RuntimeError(f"No images found in {path}")
        
        # Validate first image
        first_image = Image.open(self.image_paths[0])
        w, h = first_image.size
        if w < crop_size * 2 or h < crop_size:
            raise ValueError(
                f"Images must be at least {crop_size*2}x{crop_size} pixels. "
                f"Found image of size {w}x{h}")
            
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            input_image, real_image = load(img_path)
            
            if input_image.shape[1] < self.crop_size or input_image.shape[2] < self.crop_size:
                raise ValueError(f"Image {img_path} is too small")
            
            if self.is_train:
                input_image, real_image = random_jitter(input_image, real_image)
            else:
                input_image, real_image = resize(input_image, real_image,
                                               IMG_HEIGHT, IMG_WIDTH)
                
            input_image, real_image = normalize(input_image, real_image)
            return input_image, real_image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            raise e

class DownSample(nn.Module):
    """Downsampling block for the generator."""
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(DownSample, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
        )
        
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class UpSample(nn.Module):
    """Upsampling block for the generator."""
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UpSample, self).__init__()
        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    """Modified Generator model with adjusted architecture."""
    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()
        
        self.down_stack = nn.ModuleList([
            DownSample(input_channels, 64, apply_batchnorm=False),    # 128x128
            DownSample(64, 128),                                      # 64x64
            DownSample(128, 256),                                     # 32x32
            DownSample(256, 512),                                     # 16x16
            DownSample(512, 512),                                     # 8x8
            DownSample(512, 512),                                     # 4x4
        ])
        
        self.up_stack = nn.ModuleList([
            UpSample(512, 512, apply_dropout=True),      # 8x8
            UpSample(1024, 512, apply_dropout=True),     # 16x16
            UpSample(1024, 256),                         # 32x32
            UpSample(512, 128),                          # 64x64
            UpSample(256, 64),                           # 128x128
        ])
        
        self.last = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),  # 256x256
            nn.Tanh()
        )
        
    def forward(self, x):
        skips = []
        
        # Encoder
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        
        skips = reversed(skips[:-1])
        
        # Decoder with skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            
        x = self.last(x)
        return x

class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, 4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, stride=1, padding=0)
        )
        
    def forward(self, input_image, target_image):
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)

def generator_loss(disc_generated_output, gen_output, target):
    """Calculate generator loss."""
    loss_object = nn.BCEWithLogitsLoss()
    gan_loss = loss_object(
        disc_generated_output,
        torch.ones_like(disc_generated_output)
    )
    
    l1_loss = torch.mean(torch.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Calculate discriminator loss."""
    loss_object = nn.BCEWithLogitsLoss()
    
    real_loss = loss_object(
        disc_real_output,
        torch.ones_like(disc_real_output)
    )
    generated_loss = loss_object(
        disc_generated_output,
        torch.zeros_like(disc_generated_output)
    )
    
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def generate_images(generator, test_input, target, epoch):
    """Generate and save sample images."""
    prediction = generator(test_input)
    
    os.makedirs('training_progress', exist_ok=True)
    plt.figure(figsize=(15, 5))
    
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        img = (display_list[i].cpu().detach() + 1) * 0.5
        plt.imshow(img.permute(1, 2, 0))
        plt.axis('off')
    
    plt.savefig(f'training_progress/epoch_{epoch}.png')
    plt.close()

def train_step(generator, discriminator, input_image, target, 
               generator_optimizer, discriminator_optimizer):
    """Perform a single training step with separate backward passes for G and D."""
    
    # Train Discriminator first
    discriminator_optimizer.zero_grad()
    
    # Generate fake image
    gen_output = generator(input_image)
    
    # Real discriminator output
    disc_real_output = discriminator(input_image, target)
    # Fake discriminator output (detach gen_output to avoid generator update)
    disc_generated_output = discriminator(input_image, gen_output.detach())
    
    # Calculate discriminator loss
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    # Backward pass for discriminator
    disc_loss.backward()
    discriminator_optimizer.step()
    
    # Train Generator
    generator_optimizer.zero_grad()
    
    # Need new discriminator outputs for generator training
    # (now with the generator output not detached)
    disc_generated_output = discriminator(input_image, gen_output)
    
    # Calculate generator losses
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
        disc_generated_output, gen_output, target)
    
    # Backward pass for generator
    gen_total_loss.backward()
    generator_optimizer.step()
    
    return {
        'gen_total_loss': gen_total_loss.item(),
        'gen_gan_loss': gen_gan_loss.item(),
        'gen_l1_loss': gen_l1_loss.item(),
        'disc_loss': disc_loss.item()
    }

def print_model_summary(model, input_size):
    """Print model architecture and parameter count."""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")
    
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_size).to(next(model.parameters()).device)
            _ = model(dummy_input)
            print("Forward pass successful with shape:", input_size)
    except Exception as e:
        print(f"Forward pass failed: {str(e)}")
    finally:
        model.train()

def download_and_extract_dataset(dataset_name="facades"):
    """Download and extract the pix2pix dataset."""
    url = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'
    tar_path = f"{dataset_name}.tar.gz"
    
    print(f"Downloading {dataset_name} dataset...")
    urllib.request.urlretrieve(url, tar_path)
    
    print("Extracting dataset...")
    with tarfile.open(tar_path) as tar:
        tar.extractall()
    
    os.remove(tar_path)
    return pathlib.Path(dataset_name)

def train(dataset_path, num_epochs=200, save_interval=10):
    """Training function with enhanced error handling and diagnostics."""
    try:
        train_dataset = Pix2PixDataset(dataset_path / "train", is_train=True)
        train_loader = data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
        )
        
        test_dataset = Pix2PixDataset(
            dataset_path / "test" if (dataset_path / "test").exists() else dataset_path / "val", 
            is_train=False
        )
        test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        print(f"Dataset loaded successfully:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        raise
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Print model summaries
    print_model_summary(generator, (3, 256, 256))
    print_model_summary(discriminator, (3, 256, 256))
    
    # Initialize optimizers
    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    
    # Create directories for checkpoints and logs
    os.makedirs('checkpoints', exist_ok=True)
    writer = SummaryWriter('runs/pix2pix_training')
    
    print("\nStarting training...")
    try:
        for epoch in range(num_epochs):
            start_time = time.time()
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for i, (input_image, target) in enumerate(train_loader):
                try:
                    input_image, target = input_image.to(device), target.to(device)
                    
                    losses = train_step(
                        generator, discriminator,
                        input_image, target,
                        generator_optimizer, discriminator_optimizer
                    )
                    
                    epoch_g_loss += losses["gen_total_loss"]
                    epoch_d_loss += losses["disc_loss"]
                    
                    if i % 10 == 0:
                        print(f'Epoch {epoch}/{num_epochs} Batch {i}/{len(train_loader)} '
                              f'G_loss: {losses["gen_total_loss"]:.4f} '
                              f'D_loss: {losses["disc_loss"]:.4f}', end='\r')
                        
                        # Log to TensorBoard
                        step = epoch * len(train_loader) + i
                        writer.add_scalar('G_loss', losses["gen_total_loss"], step)
                        writer.add_scalar('D_loss', losses["disc_loss"], step)
                        
                except Exception as e:
                    print(f"\nError in batch {i}: {str(e)}")
                    continue
            
            # Epoch summary
            avg_g_loss = epoch_g_loss / len(train_loader)
            avg_d_loss = epoch_d_loss / len(train_loader)
            print(f'\nEpoch {epoch} completed in {time.time() - start_time:.2f}s')
            print(f'Average G_loss: {avg_g_loss:.4f} D_loss: {avg_d_loss:.4f}')
            
            # Save checkpoint and generate samples
            if epoch % save_interval == 0:
                # Save checkpoint
                save_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'generator_optimizer': generator_optimizer.state_dict(),
                    'discriminator_optimizer': discriminator_optimizer.state_dict(),
                    'g_loss': avg_g_loss,
                    'd_loss': avg_d_loss
                }, save_path)
                print(f"Checkpoint saved: {save_path}")
                
                # Generate sample images
                with torch.no_grad():
                    test_input, test_target = next(iter(test_loader))
                    test_input, test_target = test_input.to(device), test_target.to(device)
                    generate_images(generator, test_input, test_target, epoch)
                    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupted checkpoint
        save_path = 'checkpoints/checkpoint_interrupted.pt'
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_optimizer': generator_optimizer.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict(),
        }, save_path)
        print(f"Interrupted checkpoint saved: {save_path}")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise
        
    finally:
        writer.close()
        print("Training completed!")

if __name__ == "__main__":
    # Parse command line arguments (you can add argparse here if needed)
    dataset_name = "facades"  # Change this to use different datasets
    num_epochs = 310
    save_interval = 10
    
    try:
        print(f"Preparing to train Pix2Pix model on {dataset_name} dataset")
        
        
        # Download and prepare dataset
        dataset_path = download_and_extract_dataset(dataset_name)
        
        # Start training
        train(dataset_path, num_epochs, save_interval)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("\nCleaning up...")