#!/usr/bin/env python3
"""
Enhanced HTML Results Generator with Error Maps
Automatically generates error maps between fake_B and real_B images
and creates a comprehensive web visualization.
"""

import os
import sys
import subprocess
import shutil

def run_enhanced_generator():
    """Run the enhanced HTML generator in the appropriate results directory"""
    
    # List potential results directories
    results_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and 'results' in item.lower():
            results_dirs.append(item)
    
    if not results_dirs:
        print("❌ No results directories found!")
        return
    
    print("🔍 Found results directories:")
    for i, dir_name in enumerate(results_dirs):
        print(f"   {i+1}. {dir_name}")
    
    # Check for the specific directory that likely contains test results
    target_dirs = [d for d in results_dirs if 'pix2pix_source2target' in d]
    
    if target_dirs:
        target_dir = target_dirs[0]
        print(f"\n🎯 Using directory: {target_dir}")
    else:
        # Use the first results directory
        target_dir = results_dirs[0]
        print(f"\n🎯 Using directory: {target_dir}")
    
    # Look for test_latest subdirectory
    test_latest_dir = os.path.join(target_dir, 'test_latest')
    if os.path.exists(test_latest_dir):
        working_dir = test_latest_dir
        print(f"📁 Found test_latest directory: {working_dir}")
    else:
        working_dir = target_dir
        print(f"📁 Using base results directory: {working_dir}")
    
    # Check if images directory exists
    images_dir = os.path.join(working_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"❌ No 'images' directory found in {working_dir}")
        print("   Please make sure you're in the correct results directory.")
        return
    
    # Copy the enhanced generator to the working directory
    generator_script = 'generate_enhanced_index_with_error_maps.py'
    target_script = os.path.join(working_dir, generator_script)
    
    if os.path.exists(generator_script):
        shutil.copy2(generator_script, target_script)
        print(f"📋 Copied enhanced generator to {working_dir}")
    else:
        print(f"❌ Enhanced generator script not found: {generator_script}")
        return
    
    # Change to the working directory and run the generator
    original_dir = os.getcwd()
    try:
        os.chdir(working_dir)
        print(f"\n🚀 Running enhanced HTML generator in {working_dir}...")
        print("   This will:")
        print("   • Generate error maps between fake_B and real_B images")
        print("   • Compute SSIM, PSNR, and MSE metrics")
        print("   • Create an enhanced HTML visualization")
        print("   • Include overall performance statistics")
        
        # Run the generator
        result = subprocess.run([sys.executable, generator_script], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ HTML generation completed successfully!")
            print(f"📊 Output: {os.path.join(working_dir, 'index.html')}")
            print(f"🔥 Error maps: {os.path.join(working_dir, 'error_maps/')}")
            print("\n🌐 To view the results:")
            print(f"   firefox {os.path.join(working_dir, 'index.html')} &")
            print("   or")
            print(f"   python -m http.server 8000  # then visit http://localhost:8000")
        else:
            print(f"\n❌ Error running generator:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    print("🖼️  Enhanced Pix2Pix Results Generator with Error Analysis")
    print("=" * 60)
    run_enhanced_generator() 