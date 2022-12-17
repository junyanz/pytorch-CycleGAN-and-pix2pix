import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model


options = TrainOptions().parse()
model = create_model(options)  # create a model given opt.model and other options
model.setup(options)
