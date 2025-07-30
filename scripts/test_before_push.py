# Simple script to make sure basic usage such as training, testing, saving and loading
# runs without errors
import os
from pathlib import Path


# run a command and exit if non-zero exit status
def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == "__main__":
    # download mini datasets
    if not Path("./datasets/mini").exists():
        run("bash ./datasets/download_cyclegan_dataset.sh mini")

    if not Path("./datasets/mini_pix2pix").exists():
        run("bash ./datasets/download_cyclegan_dataset.sh mini_pix2pix")

    # pretrained cyclegan model
    if not Path("./checkpoints/horse2zebra_pretrained/latest_net_G.pth").exists():
        run("bash ./scripts/download_cyclegan_model.sh horse2zebra")
    run("python test.py --model test --dataroot ./datasets/mini --name horse2zebra_pretrained --no_dropout --num_test 1 --no_dropout")

    # pretrained pix2pix model
    if not Path("./checkpoints/facades_label2photo_pretrained/latest_net_G.pth").exists():
        run("bash ./scripts/download_pix2pix_model.sh facades_label2photo")
    if not Path("./datasets/facades").exists():
        run("bash ./datasets/download_pix2pix_dataset.sh facades")
    run("python test.py --dataroot ./datasets/facades/ --direction BtoA --model pix2pix --name facades_label2photo_pretrained --num_test 1")

    # cyclegan train/test
    run(
        "python train.py --model cycle_gan --name temp_cyclegan --dataroot ./datasets/mini --n_epochs 1 --n_epochs_decay 0 --save_latest_freq 10 --print_freq 1"
    )
    run('python test.py --model test --name temp_cyclegan --dataroot ./datasets/mini --num_test 1 --model_suffix "_A" --no_dropout')

    # pix2pix train/test
    run(
        "python train.py --model pix2pix --name temp_pix2pix --dataroot ./datasets/mini_pix2pix --n_epochs 1 --n_epochs_decay 5 --save_latest_freq 10"
    )
    run("python test.py --model pix2pix --name temp_pix2pix --dataroot ./datasets/mini_pix2pix --num_test 1")

    # template train/test
    run("python train.py --model template --name temp2 --dataroot ./datasets/mini_pix2pix --n_epochs 1 --n_epochs_decay 0 --save_latest_freq 10")
    run("python test.py --model template --name temp2 --dataroot ./datasets/mini_pix2pix --num_test 1")

    # colorization train/test (optional)
    if not Path("./datasets/mini_colorization").exists():
        run("bash ./datasets/download_cyclegan_dataset.sh mini_colorization")

    run(
        "python train.py --model colorization --name temp_color --dataroot ./datasets/mini_colorization --n_epochs 1 --n_epochs_decay 0 --save_latest_freq 5"
    )
    run("python test.py --model colorization --name temp_color --dataroot ./datasets/mini_colorization --num_test 1")
