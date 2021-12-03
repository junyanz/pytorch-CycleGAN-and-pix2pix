''''

Run using:
 $ python -m av_scene_test_map_encoder.py --dataset_mode av_scene --dataroot datasets/av_scene_data/l5kit_sample.pkl --name l5kit_sample --model av_scene --display_id 0

** To run only on CPU add: --gpu_ids -1


'''

from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
from models.base_model import BaseModel
from models import av_scene_networks
import time


class ModelTestMapEncoder(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.map_enc = av_scene_networks.MapEncoder(opt)
        self.loss_criterion = torch.nn.L1Loss()
        print('Map encoder parameters: ', [p[0] for p in self.map_enc.named_parameters()])
        self.optimizer = torch.optim.Adam(self.map_enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer)

    def set_input(self, map_feat):
        self.map_feat = map_feat
        self.ground_truth = None

    def forward(self):
        self.prediction = None

    def backward(self):
        self.loss = self.loss_criterion(self.prediction, self.ground_truth)
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

if __name__ == '__main__':

    n_epoch = 2

    train_opt = TrainOptions().parse()  # get training options
    train_dataset = create_dataset(train_opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    model = create_model(train_opt)  # create a model given opt.model and other options
    model.setup(train_opt)  # regular setup: load and print networks; create schedulers

    ##########
    # Train
    #########
    start_time = time.time()
    for i_epoch in range(n_epoch):
        for i_batch, data in enumerate(train_dataset):
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.update_learning_rate()  # update learning rates *after* first step (https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
        print(f'End of epoch {i_epoch}, elapsed time {time.time() - start_time}')

    ##########
    # Test
    ##########
    model.eval()
    test_opt = TestOptions().parse()  # get test options
    eval_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    for i, data in enumerate(eval_dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
