#!/usr/bin/env python3
"""
Simple CycleGAN inference examples
"""

import numpy as np
from PIL import Image
from simple_cyclegan_inference import (
    SimpleCycleGANInference, 
    simple_cyclegan_inference, 
    load_cyclegan_model, 
    generate_cyclegan_image,
    save_image
)

def create_test_image():
    """Create a simple test ultrasound-like image"""
    # Create a test image that looks like ultrasound
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add some ultrasound-like patterns
    image[50:200, 50:200] = 80   # Background
    image[100:150, 100:150] = 150  # Bright region
    image[120:130, 120:130] = 200  # Very bright spot
    
    # Add some noise
    noise = np.random.randint(-20, 20, image.shape)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image

def example_1_synthetic_to_real():
    """Example 1: Convert synthetic ultrasound to real ultrasound"""
    print("=== Example 1: Synthetic → Real ===")
    
    checkpoint_dir = "checkpoints/synt2realUS_grf_phantom_busi_v2"
    
    # Create synthetic-like test image
    synthetic_image = create_test_image()
    save_image(synthetic_image, "test_synthetic.png")
    print("✓ Created test synthetic image")
    
    try:
        # Convert synthetic to real (AtoB direction)
        real_image = simple_cyclegan_inference(
            checkpoint_dir, 
            synthetic_image, 
            direction='AtoB'
        )
        
        print(f"✓ Generated real image, shape: {real_image.shape}")
        save_image(real_image, "generated_real.png")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def example_2_real_to_synthetic():
    """Example 2: Convert real ultrasound to synthetic ultrasound"""
    print("\n=== Example 2: Real → Synthetic ===")
    
    checkpoint_dir = "checkpoints/synt2realUS_grf_phantom_busi_v2"
    
    # Create real-like test image (more textured)
    real_image = create_test_image()
    # Add more texture to make it "real-like"
    for i in range(5):
        noise = np.random.randint(-10, 10, real_image.shape)
        real_image = np.clip(real_image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    save_image(real_image, "test_real.png")
    print("✓ Created test real image")
    
    try:
        # Convert real to synthetic (BtoA direction)
        synthetic_image = simple_cyclegan_inference(
            checkpoint_dir, 
            real_image, 
            direction='BtoA'
        )
        
        print(f"✓ Generated synthetic image, shape: {synthetic_image.shape}")
        save_image(synthetic_image, "generated_synthetic.png")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def example_3_reusable_model():
    """Example 3: Reusable model for multiple images"""
    print("\n=== Example 3: Reusable model ===")
    
    checkpoint_dir = "checkpoints/synt2realUS_grf_phantom_busi_v2"
    
    try:
        # Load model once
        model = load_cyclegan_model(checkpoint_dir, direction='AtoB')
        
        # Process multiple images
        for i in range(3):
            # Create different test images
            test_image = create_test_image()
            # Add variation
            test_image = test_image + np.random.randint(-30, 30, test_image.shape)
            test_image = np.clip(test_image, 0, 255).astype(np.uint8)
            
            # Generate result
            result = generate_cyclegan_image(model, test_image)
            
            print(f"✓ Processed image {i+1}, shape: {result.shape}")
            save_image(test_image, f"input_{i+1}.png")
            save_image(result, f"output_{i+1}.png")
            
    except Exception as e:
        print(f"✗ Error: {e}")

def example_4_from_file():
    """Example 4: Load and process from image file"""
    print("\n=== Example 4: From image file ===")
    
    checkpoint_dir = "checkpoints/synt2realUS_grf_phantom_busi_v2"
    
    # Create and save test image
    test_image = create_test_image()
    Image.fromarray(test_image).save("input_file.png")
    print("✓ Created input_file.png")
    
    try:
        # Process from file path
        result = simple_cyclegan_inference(
            checkpoint_dir, 
            "input_file.png", 
            direction='AtoB'
        )
        
        print(f"✓ Generated image from file, shape: {result.shape}")
        save_image(result, "output_from_file.png")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def check_available_models():
    """Check available CycleGAN models"""
    print("=== Available CycleGAN Models ===")
    
    import os
    checkpoints_dir = "checkpoints"
    
    if os.path.exists(checkpoints_dir):
        models = [d for d in os.listdir(checkpoints_dir) 
                 if os.path.isdir(os.path.join(checkpoints_dir, d))]
        
        print("Available models:")
        for model in models:
            model_path = os.path.join(checkpoints_dir, model)
            generators = [f for f in os.listdir(model_path) 
                         if f.startswith('latest_net_G')]
            print(f"  - {model}")
            for gen in generators:
                print(f"    {gen}")
    else:
        print("No checkpoints directory found")

if __name__ == "__main__":
    print("Simple CycleGAN Inference - Test Examples")
    print("=" * 50)
    
    # Check available models
    check_available_models()
    
    # Run examples
    example_1_synthetic_to_real()
    example_2_real_to_synthetic()
    example_3_reusable_model()
    example_4_from_file()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Check the generated image files")
    print("\nUsage Summary:")
    print("- AtoB: Synthetic → Real (use G_A)")
    print("- BtoA: Real → Synthetic (use G_B)") 