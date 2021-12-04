''''

Run using:
 $ python -m avsg_run_check_map_enc.py --dataset_mode avsg  --model avsg_check_map_enc --dataroot datasets/avsg_data/l5kit_train.pkl --data_eval datasets/avsg_data/l5kit_validation.pkl
* To change dataset files change --dataroot and --data_eval
* To run only on CPU add: --gpu_ids -1
* Name the experiment with --name

'''
import os
import time
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions


if __name__ == '__main__':

    n_epoch = 1
    opt = TrainOptions(is_image_data=False).parse()  # get training options
    assert os.path.isfile(opt.data_eval)
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    ##########
    # Train
    #########
    start_time = time.time()
    for i_epoch in range(n_epoch):
        for i_batch, data in enumerate(train_dataset):
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights
            model.update_learning_rate()  # update learning rates *after* first step (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
        print(f'End of epoch {i_epoch}, elapsed time {time.time() - start_time}')

    ##########
    # Test
    ##########
    model.eval()
    opt.dataroot = opt.data_eval
    eval_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(eval_dataset)  # get the number of images in the dataset.
    print('The number of test samples = %d' % dataset_size)
    for i, data in enumerate(eval_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
