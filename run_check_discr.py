''''

Run using:
 $ python -m run_check_map_enc.py
 --dataset_mode avsg  --model avsg_check_discr --dataroot datasets/avsg_data/l5kit_train.pkl --data_eval datasets/avsg_data/l5kit_validation.pkl
* To change dataset files change --dataroot and --data_eval
* To run only on CPU add: --gpu_ids -1
* To limit the datasets size --max_dataset_size 1000
* Name the experiment with --name

'''
import time

import wandb

from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from models.helper_func import run_validation
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions(is_image_data=False).parse()  # get training options
    assert opt.model == 'avsg_check_discr'

    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    opt.dataroot = opt.data_eval
    eval_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    eval_dataset_size = len(eval_dataset)  # get the number of images in the dataset.
    print('The number of test samples = %d' % eval_dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    n_epochs = opt.n_epochs
    print('Number of training epochs: ', n_epochs)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    ##########
    # Train
    #########
    iter_print_freq = 20
    start_time = time.time()
    total_iter = 0
    for i_epoch in range(n_epochs):
        for i_batch, data in enumerate(train_dataset):
            # unpack data from dataset and apply preprocessing
            is_valid = model.set_input(data)
            if not is_valid:
                # if the data sample is not valid to use
                continue
            # calculate loss functions, get gradients, update network weights
            model.optimize_parameters()
            # update learning rates *after* first step (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
            model.update_learning_rate()
            if total_iter % iter_print_freq == 0:
                model.set_input(data)
                model.forward()  # run inference
                loss = model.loss_criterion(model.prediction, model.ground_truth)
                val_loss = run_validation(model, eval_dataset, n_batches=1)
                print(f'Epoch {i_epoch}, batch {i_batch}, total_iter {total_iter},  train batch loss {loss:.2},'
                      f' val batch loss {val_loss:.2}')
                if opt.use_wandb:
                    wandb.log({'train_batch_loss': loss, 'val_batch_loss':  val_loss, 'epoch': i_epoch})
            total_iter += 1
        print(f'End of epoch {i_epoch}, elapsed time {(time.time() - start_time):.2f}')

    ##########
    # Test
    ##########
    loss_avg = run_validation(model, eval_dataset)
    print(f'Average test loss over  {eval_dataset_size} samples is loss {loss_avg:.2f}')
