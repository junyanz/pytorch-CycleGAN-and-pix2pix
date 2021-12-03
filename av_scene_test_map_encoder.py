''''

Run using:
 $ python -m av_scene_test_map_encoder.py --dataset_mode av_scene --dataroot datasets/av_scene_data/l5kit_sample.pkl --name l5kit_sample --model av_scene --display_id 0

** To run only on CPU add: --gpu_ids -1


'''

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
from models.base_model import BaseModel
from models import networks
from models import av_scene_networks
import time


class ModelTestMapEncoder(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.map_enc = av_scene_networks.MapEncoder(opt)
        self.criterionL1 = torch.nn.L1Loss()
        print('Map encoder parameters: ', [p[0] for p in self.map_enc.named_parameters()])
        self.optimizer_prediction = torch.optim.Adam(self.map_enc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer_prediction)


if __name__ == '__main__':

    n_epoch = 2

    opt = TrainOptions().parse()  # get training options
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(train_dataset)  # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    ##########
    # Train
    #########
    total_iters = 0  # the total number of training iterations
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
    eval_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    for i, data in enumerate(eval_dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,
                    use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML


