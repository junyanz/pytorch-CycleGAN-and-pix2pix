"""
CycleGAN inference script for Jupyter notebooks
This allows you to set options programmatically instead of using command line arguments
"""

import os
import sys
from argparse import Namespace
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

# Add current directory to path
sys.path.append('.')

def create_test_options(
    dataroot='datasets/synt2realUS/testB',
    name='synt2realUS_grf_phantom_busi_v2',
    model='test',
    input_nc=1,
    output_nc=1,
    ngf=64,
    ndf=64,
    netG='resnet_9blocks',
    netD='basic',
    no_dropout=True,
    norm='instance',
    init_type='normal',
    init_gain=0.02,
    gpu_ids=[0],
    checkpoints_dir='./checkpoints',
    results_dir='./results',
    phase='test',
    epoch='latest',
    load_iter=0,
    verbose=False,
    suffix='',
    aspect_ratio=1.0,
    display_winsize=256,
    crop_size=256,
    load_size=256,
    dataset_mode='single',
    direction='AtoB',
    serial_batches=True,
    no_flip=True,
    display_id=-1,
    num_test=50,
    num_threads=0,
    batch_size=1,
    max_dataset_size=float("inf"),
    preprocess='resize_and_crop',
    use_wandb=False,
    wandb_project_name='CycleGAN-and-pix2pix',
    eval=False
):
    """
    Create test options programmatically (for notebook use)
    
    Args:
        dataroot: Path to test images
        name: Name of the experiment (should match training)
        model: Model type ('test' for single direction, 'cycle_gan' for both)
        input_nc: Number of input channels
        output_nc: Number of output channels
        no_dropout: Whether to use dropout
        gpu_ids: List of GPU IDs to use
        ... (other standard CycleGAN options)
    
    Returns:
        Namespace object with all options
    """
    
    opt = Namespace()
    
    # Basic options
    opt.dataroot = dataroot
    opt.name = name
    opt.model = model
    opt.input_nc = input_nc
    opt.output_nc = output_nc
    opt.ngf = ngf
    opt.ndf = ndf
    opt.netG = netG
    opt.netD = netD
    opt.no_dropout = no_dropout
    opt.norm = norm
    opt.init_type = init_type
    opt.init_gain = init_gain
    opt.gpu_ids = gpu_ids
    opt.checkpoints_dir = checkpoints_dir
    opt.results_dir = results_dir
    opt.phase = phase
    opt.epoch = epoch
    opt.load_iter = load_iter
    opt.verbose = verbose
    opt.suffix = suffix
    
    # Display options
    opt.aspect_ratio = aspect_ratio
    opt.display_winsize = display_winsize
    opt.display_id = display_id
    
    # Dataset options
    opt.crop_size = crop_size
    opt.load_size = load_size
    opt.dataset_mode = dataset_mode
    opt.direction = direction
    opt.serial_batches = serial_batches
    opt.no_flip = no_flip
    opt.num_test = num_test
    opt.num_threads = num_threads
    opt.batch_size = batch_size
    opt.max_dataset_size = max_dataset_size
    opt.preprocess = preprocess
    
    # Wandb options
    opt.use_wandb = use_wandb
    opt.wandb_project_name = wandb_project_name
    
    # Evaluation
    opt.eval = eval
    
    # IMPORTANT: Add missing attributes that CycleGAN framework expects
    opt.isTrain = False  # This is the missing attribute causing the error
    opt.continue_train = False
    opt.epoch_count = 1
    opt.niter = 100
    opt.niter_decay = 100
    opt.beta1 = 0.5
    opt.lr = 0.0002
    opt.gan_mode = 'lsgan'
    opt.pool_size = 50
    opt.lr_policy = 'linear'
    opt.lr_decay_iters = 50
    opt.lambda_A = 10.0
    opt.lambda_B = 10.0
    opt.lambda_identity = 0.5
    opt.save_latest_freq = 5000
    opt.save_epoch_freq = 5
    opt.save_by_iter = False
    opt.print_freq = 100
    opt.update_html_freq = 1000
    opt.display_freq = 400
    opt.display_ncols = 4
    opt.display_server = "http://localhost"
    opt.display_env = "main"
    opt.display_port = 8097
    opt.n_layers_D = 3
    opt.ndf = ndf
    opt.netD = netD
    opt.n_layers_D = 3
    opt.no_html = False
    
    # Additional missing attributes
    opt.model_suffix = ''  # This was the missing attribute causing the new error
    opt.dataset_mode = dataset_mode
    opt.serial_batches = serial_batches
    opt.no_flip = no_flip
    opt.load_size = load_size
    opt.crop_size = crop_size
    opt.max_dataset_size = max_dataset_size
    opt.preprocess = preprocess
    opt.no_flip = no_flip
    opt.display_id = display_id
    
    # Model-specific options that might be needed
    opt.netG_A = netG
    opt.netG_B = netG
    opt.netD_A = netD
    opt.netD_B = netD
    opt.lambda_identity = 0.5
    opt.lambda_A = 10.0
    opt.lambda_B = 10.0
    
    # Additional training options (even though we're testing)
    opt.lr_decay_iters = 50
    opt.beta1 = 0.5
    opt.lr = 0.0002
    opt.lr_policy = 'linear'
    opt.gan_mode = 'lsgan'
    opt.pool_size = 50
    
    # Additional display options
    opt.display_ncols = 4
    opt.display_winsize = display_winsize
    opt.display_freq = 400
    opt.update_html_freq = 1000
    opt.print_freq = 100
    opt.display_server = "http://localhost"
    opt.display_env = "main"
    opt.display_port = 8097
    
    # Save options
    opt.save_latest_freq = 5000
    opt.save_epoch_freq = 5
    opt.save_by_iter = False
    opt.no_html = False
    
    # Set device
    if len(gpu_ids) > 0 and gpu_ids[0] >= 0:
        opt.device = f'cuda:{gpu_ids[0]}'
    else:
        opt.device = 'cpu'
        opt.gpu_ids = []
    
    return opt

def run_cyclegan_test(opt):
    """
    Run CycleGAN test with the given options
    
    Args:
        opt: Options namespace (from create_test_options)
    
    Returns:
        model: The loaded model
        dataset: The test dataset
        webpage: HTML webpage for results
    """
    
    print("Creating dataset...")
    dataset = create_dataset(opt)
    print(f"Dataset created with {len(dataset)} images")
    
    print("Creating model...")
    model = create_model(opt)
    model.setup(opt)
    print("Model created and loaded")
    
    # Initialize wandb if requested
    if opt.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=opt.wandb_project_name, 
                name=opt.name, 
                config=opt
            ) if not wandb.run else wandb.run
            wandb_run._label(repo='CycleGAN-and-pix2pix')
        except ImportError:
            print('Warning: wandb package not found. Skipping wandb logging.')
            opt.use_wandb = False
    
    # Create results directory
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'
    
    print(f'Creating web directory: {web_dir}')
    webpage = html.HTML(web_dir, f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')
    
    # Run inference
    if opt.eval:
        model.eval()
    
    print("Running inference...")
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        if i % 5 == 0:
            print(f'Processing ({i:04d})-th image... {img_path}')
        
        save_images(
            webpage, 
            visuals, 
            img_path, 
            aspect_ratio=opt.aspect_ratio, 
            width=opt.display_winsize, 
            use_wandb=opt.use_wandb
        )
    
    webpage.save()
    print(f"Results saved to {web_dir}")
    
    return model, dataset, webpage

# Convenience function for your specific use case
def run_synt2real_test(
    test_dir='datasets/synt2realUS/testB',
    model_name='synt2realUS_grf_phantom_busi_v2',
    num_test=50,
    use_wandb=False,
    gpu_id=0
):
    """
    Convenience function for your specific synthetic to real ultrasound test
    
    Args:
        test_dir: Directory with test images
        model_name: Name of your trained model
        num_test: Number of images to test
        use_wandb: Whether to log to wandb
        gpu_id: GPU ID to use
    
    Returns:
        Results from run_cyclegan_test
    """
    
    opt = create_test_options(
        dataroot=test_dir,
        name=model_name,
        model='test',
        input_nc=1,
        output_nc=1,
        no_dropout=True,
        num_test=num_test,
        use_wandb=use_wandb,
        gpu_ids=[gpu_id]
    )
    
    return run_cyclegan_test(opt)

if __name__ == "__main__":
    # Example usage
    print("CycleGAN Notebook Inference Ready!")
    print("\nExample usage:")
    print("opt = create_test_options(")
    print("    dataroot='datasets/synt2realUS/testB',")
    print("    name='synt2realUS_grf_phantom_busi_v2',")
    print("    model='test',")
    print("    input_nc=1,")
    print("    output_nc=1,")
    print("    no_dropout=True,")
    print("    use_wandb=False")
    print(")")
    print("model, dataset, webpage = run_cyclegan_test(opt)")