"""General training script for image-to-image translation

Example:
    Train a CycleGAN model:
        python train --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

Options:
    Available models ('--model')
        pix2pix, cyclegan, colorization

    Available datasets ('--dataset_mode')
       aligned, unaligned, single, colorization

Required parameters:
    You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

Features:
    Loads networks, dataset, etc
    During the training, it also visualize/save the images, print/save the loss plot, and save models.
    The script supports continue/resume training. Use '--continue_train' to resume your previous training.

References:
    See options/base_options.py and options/train_options.py for more training options.
    See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
    See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer

if __name__ == '__main__':
    # Get training options
    opt = TrainOptions().parse()

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)  # regular stuff; load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualiser that displays/saves images and plots

    total_iters = 0

    # Apparently training is easy enough that we'll do it inline!
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading

        epoch_iter = 0
        visualizer.reset()

        for i, data in enumerate(dataset):
            iter_start_time = time.time()  # timer for computation

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            # this is equivalent to metrics = train(x,y) in the BigGAN code
            model.set_input(data)  # unpack from the dataset
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0

                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))

                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))

            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.
