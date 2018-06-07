# Simple script to make sure basic usage
# such as training, testing, saving and loading
# runs without errors.
import os


def run_bash_command(command):
    print(command)
    exit_status = os.system(command)    
    if exit_status > 0:
        exit(1)

        
if __name__ == '__main__':
    if not os.path.exists('datasets/mini'):
        run_bash_command('bash datasets/download_cyclegan_dataset.sh mini')

    if not os.path.exists('datasets/mini_pix2pix'):
        run_bash_command('bash datasets/download_cyclegan_dataset.sh mini_pix2pix')

    # pretrained
    if not os.path.exists('./checkpoints/horse2zebra_pretrained/latest_net_G.pth'):
        run_bash_command('bash pretrained_models/download_cyclegan_model.sh horse2zebra')
    run_bash_command('python test.py --model test --dataroot ./datasets/mini --name horse2zebra_pretrained --no_dropout --how_many 1')

    # test cyclegan
    run_bash_command('python train.py --name temp --dataroot ./datasets/mini --niter 1 --niter_decay 0 --save_latest_freq 10  --display_freq 1')
    run_bash_command('python test.py --name temp --dataroot ./datasets/mini --how_many 1')

    # test pix2pix
    run_bash_command('python train.py --model pix2pix --name temp --dataroot ./datasets/mini_pix2pix --niter 1 --niter_decay 0 --save_latest_freq 10')
    run_bash_command('python test.py --model pix2pix --name temp --dataroot ./datasets/mini_pix2pix --how_many 1 --which_direction BtoA')
    
