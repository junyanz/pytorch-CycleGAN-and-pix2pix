# Simple script to make sure basic usage
# such as training, testing, saving and loading
# runs without errors.
import os


def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    if not os.path.exists('./datasets/mini'):
        run('bash ./datasets/download_cyclegan_dataset.sh mini')

    if not os.path.exists('./datasets/mini_pix2pix'):
        run('bash ./datasets/download_cyclegan_dataset.sh mini_pix2pix')

    # pretrained cyclegan model
    if not os.path.exists('./checkpoints/horse2zebra_pretrained/latest_net_G.pth'):
        run('bash ./scripts/download_cyclegan_model.sh horse2zebra')
    run('python test.py --model test --dataroot ./datasets/mini --name horse2zebra_pretrained --no_dropout --how_many 1')

    # pretrained pix2pix model
    if not os.path.exists('./checkpoints/facades_label2photo_pretrained/latest_net_G.pth'):
        run('bash ./scripts/download_pix2pix_model.sh facades_label2photo')
    if not os.path.exists('./datasets/facades'):
        run('bash ./datasets/download_pix2pix_dataset.sh facades')
    run('python test.py --dataroot ./datasets/facades/ --which_direction BtoA --model pix2pix --name facades_label2photo_pretrained --how_many 1')

    # cyclegan train/test
    run('python train.py --name temp --dataroot ./datasets/mini --niter 1 --niter_decay 0 --save_latest_freq 10  --print_freq 1 --display_id -1')
    run('python test.py --name temp --dataroot ./datasets/mini --how_many 1 --model_suffix "_A"')

    # pix2pix train/test
    run('python train.py --model pix2pix --name temp --dataroot ./datasets/mini_pix2pix --niter 1 --niter_decay 0 --save_latest_freq 10 --display_id -1')
    run('python test.py --model pix2pix --name temp --dataroot ./datasets/mini_pix2pix --how_many 1 --which_direction BtoA')
