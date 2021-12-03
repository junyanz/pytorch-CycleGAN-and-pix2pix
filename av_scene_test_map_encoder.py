''''

Run using:
 $ python -m av_scene_test_map_encoder.py --dataset_mode av_scene --dataroot datasets/av_scene_data/l5kit_sample.pkl --name l5kit_sample --model av_scene --display_id 0

** To run only on CPU add: --gpu_ids -1


'''

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations