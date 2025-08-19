import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

# Add the current directory to path to import models
sys.path.append('.')
from models import networks

class SimpleCycleGANInference:
    """Simple CycleGAN inference pipeline"""
    
    def __init__(self, checkpoint_dir, direction='AtoB', device='cuda:0'):
        """
        Initialize CycleGAN inference
        
        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., 'checkpoints/synt2realUS_grf_phantom_busi_v2')
            direction: 'AtoB' or 'BtoA' (which generator to use)
            device: Device to run on
        """
        self.device = device
        self.direction = direction
        self.checkpoint_dir = checkpoint_dir
        
        # Load generator
        self.generator = self.load_generator()
        
        # Set up transforms
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        self.inverse_transform = transforms.Compose([
            transforms.Normalize((-1,), (2,)),  # Denormalize from [-1, 1] to [0, 1]
            transforms.ToPILImage()
        ])
    
    def load_generator(self):
        """Load the generator model"""
        print(f"Loading CycleGAN generator from: {self.checkpoint_dir}")
        
        # Determine which generator to load
        if self.direction == 'AtoB':
            generator_path = os.path.join(self.checkpoint_dir, 'latest_net_G_A.pth')
        else:  # BtoA
            generator_path = os.path.join(self.checkpoint_dir, 'latest_net_G_B.pth')
        
        if not os.path.exists(generator_path):
            raise FileNotFoundError(f"Generator not found: {generator_path}")
        
        # Create generator network
        # Using default CycleGAN architecture
        generator = networks.define_G(
            input_nc=1,      # Input channels (grayscale)
            output_nc=1,     # Output channels (grayscale)
            ngf=64,          # Number of filters in generator
            netG='resnet_9blocks',  # Generator architecture
            norm='instance', # Normalization type
            use_dropout=False,
            init_type='normal',
            init_gain=0.02,
            gpu_ids=[int(self.device.split(':')[1])] if 'cuda' in self.device else []
            # there should be some options to load the model from the checkpoint directory 
        )
        
        # Load weights
        checkpoint = torch.load(generator_path, map_location=self.device)
        generator.load_state_dict(checkpoint)
        generator.eval()
        
        print(f"✓ Generator loaded successfully! ({self.direction})")
        return generator
    
    def preprocess_image(self, image):
        """Preprocess input image"""
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply transforms
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def generate(self, input_image):
        """
        Generate translated image
        
        Args:
            input_image: Input image (PIL Image, numpy array, or path string)
            
        Returns:
            Generated image as numpy array (0-255, uint8)
        """
        # Load image if path is provided
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Image not found: {input_image}")
            input_image = Image.open(input_image)
        
        # Preprocess
        input_tensor = self.preprocess_image(input_image)
        
        # Generate
        with torch.no_grad():
            generated_tensor = self.generator(input_tensor)
        
        # Post-process
        generated_tensor = generated_tensor.squeeze(0)  # Remove batch dimension
        generated_image = self.inverse_transform(generated_tensor)
        
        # Convert to numpy array
        result = np.array(generated_image)
        
        return result

# Simple function interface
def load_cyclegan_model(checkpoint_dir, direction='AtoB', device='cuda:0'):
    """Load CycleGAN model - simple function"""
    return SimpleCycleGANInference(checkpoint_dir, direction, device)

def generate_cyclegan_image(model, input_image):
    """Generate image with CycleGAN - simple function"""
    return model.generate(input_image)

def simple_cyclegan_inference(checkpoint_dir, input_image, direction='AtoB', device='cuda:0'):
    """
    One-line CycleGAN inference
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        input_image: Input image (path, PIL Image, or numpy array)
        direction: 'AtoB' or 'BtoA'
        device: Device to run on
        
    Returns:
        Generated image as numpy array (0-255, uint8)
    """
    model = SimpleCycleGANInference(checkpoint_dir, direction, device)
    return model.generate(input_image)

def save_image(image_array, output_path):
    """Save numpy array as image"""
    if isinstance(image_array, np.ndarray):
        Image.fromarray(image_array).save(output_path)
    else:
        image_array.save(output_path)
    print(f"✓ Image saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    checkpoint_dir = "checkpoints/synt2realUS_grf_phantom_busi_v2"
    
    print("Simple CycleGAN Inference Pipeline Ready!")
    print("\nUsage Examples:")
    print("1. One-line inference:")
    print(f"   result = simple_cyclegan_inference('{checkpoint_dir}', 'input.jpg', 'AtoB')")
    print("\n2. Reusable model:")
    print(f"   model = load_cyclegan_model('{checkpoint_dir}', 'AtoB')")
    print("   result = generate_cyclegan_image(model, 'input.jpg')")
    print("\n3. Class-based:")
    print(f"   inferencer = SimpleCycleGANInference('{checkpoint_dir}', 'AtoB')")
    print("   result = inferencer.generate('input.jpg')")
    print("\nDirections:")
    print("  'AtoB': Synthetic → Real (G_A)")
    print("  'BtoA': Real → Synthetic (G_B)") 