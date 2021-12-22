"""General-purpose training script for image-to-image translation.

* To run training:
$ python -m avsg_train
 --dataset_mode avsg  --model avsg --dataroot datasets/avsg_data/l5kit_sample.pkl

* Replace l5kit_sample.pkl with l5kit_train.pkl or l5kit_train_full.pkl for larger datasets
* To run only on CPU add: --gpu_ids -1
* To limit the datasets size --max_dataset_size 1000
* Name the experiment with --name

* Run visdom before training by $ python -m visdom.server
Or you can also disable the visdom by setting: --display_id 0


This script works for various models (with option '--model') and
different datasets (with option '--dataset_mode').
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.


See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch

if __name__ == '__main__':
    run_start_time = time.time()
    opt = TrainOptions(is_image_data=False).parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    opt.device = model.device
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    t_data = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1
            epoch_iter += 1

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time


            # unpack data from dataset and apply preprocessing:
            is_valid = model.set_input(data)
            if not is_valid:
                # if the data sample is not valid to use
                continue

            # display images on visdom and save images to an HTML file:
            if total_iters == 1 or total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                visuals_dict, wandb_logs = model.get_visual_samples(dataset, opt, epoch, epoch_iter, run_start_time)
                visualizer.display_current_results(visuals_dict, epoch, epoch_iter, save_result,
                                                   file_type='jpg', wandb_logs=wandb_logs)
                print(f'Figure saved. epoch #{epoch}, epoch_iter #{epoch_iter}, total_iter #{total_iters}')

            # calculate loss functions, get gradients, update network weights:
            model.optimize_parameters()

            # update learning rates (must be after first model update step):
            model.update_learning_rate()

            # print training losses and save logging information to the disk:
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # cache our latest model every <save_latest_freq> iterations:
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # cache our model every <save_epoch_freq> epochs:
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
