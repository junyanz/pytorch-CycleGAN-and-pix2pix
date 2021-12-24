''''

Run using:
 $ python -m avsg_run_check_map_enc.py
 --dataset_mode avsg  --model avsg_check_map_enc --dataroot datasets/avsg_data/l5kit_train.pkl --data_eval datasets/avsg_data/l5kit_validation.pkl
* To change dataset files change --dataroot and --data_eval
* To run only on CPU add: --gpu_ids -1
* To limit the datasets size --max_dataset_size 1000
* Name the experiment with --name

'''
import os
import time
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions


if __name__ == '__main__':


    opt = TrainOptions(is_image_data=False).parse()  # get training options
    n_epoch = opt.epoch_count
    print('Number of training epochs: ', n_epoch)
    assert os.path.isfile(opt.data_eval)
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    ##########
    # Train
    #########
    iter_print_freq = 20
    start_time = time.time()
    total_iter = 0
    for i_epoch in range(n_epoch):
        for i_batch, data in enumerate(train_dataset):
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            model.update_learning_rate()  # update learning rates *after* first step (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
            if total_iter % iter_print_freq == 0:
                model.set_input(data)
                model.forward()  # run inference
                loss = model.loss_criterion(model.prediction, model.ground_truth)
                print(f'Epoch {i_epoch}, batch {i_batch}, total_iter {total_iter},  loss {loss}')
            total_iter += 1
        print(f'End of epoch {i_epoch}, elapsed time {time.time() - start_time}')

    ##########
    # Test
    ##########
    model.eval()
    opt.dataroot = opt.data_eval
    eval_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(eval_dataset)  # get the number of images in the dataset.
    print('The number of test samples = %d' % dataset_size)
    n_loss_calc = 0
    loss_sum = 0
    for i, data in enumerate(eval_dataset):
        model.set_input(data)  # unpack data from data loader
        model.forward()  # run inference
        loss = model.loss_criterion(model.prediction, model.ground_truth)
        loss_sum += loss
        n_loss_calc += 1
    loss_avg = loss_sum / n_loss_calc
    print(f'Average test loss over  {dataset_size} samples is loss {loss_avg}')
