import numpy as np
import sys
import ntpath
import time
from . import util, html
from pathlib import Path
import wandb
import os
import torch.distributed as dist


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    name = Path(image_path[0]).stem

    webpage.add_header(name)
    ims, txts, links = [], [], []
    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = f"{name}_{label}.png"
        save_path = image_dir / image_name
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses wandb for logging (optional) and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: Initialize wandb (if enabled)
        Step 3: create an HTML object for saving HTML files
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.use_wandb = opt.use_wandb
        self.current_epoch = 0

        # Initialize wandb if enabled
        if self.use_wandb:
            # Only initialize wandb on main process (rank 0)
            if not dist.is_initialized() or dist.get_rank() == 0:
                self.wandb_project_name = getattr(opt, "wandb_project_name", "CycleGAN-and-pix2pix")
                self.wandb_run = wandb.init(project=self.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
                self.wandb_run._label(repo="CycleGAN-and-pix2pix")
            else:
                self.wandb_run = None

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = Path(opt.checkpoints_dir) / opt.name / "web"
            self.img_dir = self.web_dir / "images"
            print(f"create web directory {self.web_dir}...")
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = Path(opt.checkpoints_dir) / opt.name / "loss_log.txt"
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(f"================ Training Loss ({now}) ================\n")

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def set_dataset_size(self, dataset_size):
        """Set the dataset size for global step calculation"""
        self.dataset_size = dataset_size

    def _calculate_global_step(self, epoch, epoch_iter):
        """Calculate global step from epoch and epoch_iter"""
        # Assuming epoch starts from 1 and epoch_iter is cumulative within epoch
        return (epoch - 1) * self.dataset_size + epoch_iter

    def display_current_results(self, visuals, epoch: int, total_iters: int, save_result=False):
        """Save current results to wandb and HTML file."""
        # Only display results on main process (rank 0)
        if "LOCAL_RANK" in os.environ and dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            ims_dict = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_image = wandb.Image(image_numpy, caption=f"{label} - Step {total_iters}")
                ims_dict[f"results/{label}"] = wandb_image
            self.wandb_run.log(ims_dict, step=total_iters)

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = self.img_dir / f"epoch{epoch:03d}_{label}.png"
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, f"Experiment name = {self.name}", refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header(f"epoch [{n}]")
                ims, txts, links = [], [], []

                for label, image in visuals.items():
                    img_path = f"epoch{n:03d}_{label}.png"
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, total_iters, losses):
        """Log current losses to wandb

        Parameters:
            total_iters (int)     -- current training iteration during this epoch
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        # Only plot losses on main process (rank 0)
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.use_wandb:
            self.wandb_run.log(losses, step=total_iters)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        message = f"[Rank {local_rank}] (epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) "
        for k, v in losses.items():
            message += f", {k}: {v:.3f}"
        message += "\n"
        print(message)  # print the message on ALL ranks with rank info

        # Only save to log file on main process (rank 0)
        if local_rank == 0:
            with open(self.log_name, "a") as log_file:
                log_file.write(f"{message}\n")  # save the message
