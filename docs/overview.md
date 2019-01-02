## Overview of Code Structure
We give a brief overview of each directory and each file. Please see the documentation in each file for more details. If you have questions, you may find useful information in [training/test tips](tips.md) and [frequently asked questions](qa.md).

[train.py](../train.py) is a general-purpose training script. It works for various models (with option `--model`: e.g., `pix2pix`, `cyclegan`, `colorization`) and different datasets (with option `--dataset_mode`: e.g., `aligned`, `unaligned`, `single`, `colorization`). See the main [README](.../README.md) and Training/test [tips](tips.md) for more details.

[test.py](../test.py) is a general-purpose test script. Once you have trained your model with `train.py`, you can use this script to test the model. It will load a saved model from `--checkpoints_dir` and save the results to `--results_dir`. See the main [README](.../README.md) and Training/test [tips](tips.md) for more details.


[data](../data) directory contains all the modules related to data loading and data preprocessing.
* [\_\_init\_\_.py](../data/__init__.py) implements the interface between this package and training/test script. In the `train.py` and `test.py`, we call `from data import CreateDataLoader` and `data_loader = CreateDataLoader(opt)` to create a dataloader given the option `opt`.
* [base_dataset.py](../data/base_dataset.py) implements an abstract base class for datasets. It also includes common transformation functions `get_transform` and `get_simple_transform` which can be used in subclasses. To add a custom dataset class called `dummy`, you need to add a file called `dummy_dataset.py` and define a subclass `DummyDataset` inherited from `BaseDataset`. You need to implement four functions: `name`, `__len__`, `__getitem__`, and optionally `modify_commandline_options`. You can then use this dataset class by specifying flag `--dataset_mode dummy`.
* [image_folder.py](../data/image_folder.py) implements an image folder class. We modify the official PyTorch image folder [code](https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py) so that this class can load images from both the current directory and its subdirectories.
* [template_dataset.py](../data/template_dataset.py) provides a dataset class template with detailed documentation. Check out this file if you plan to implement your own dataset class.
* [aligned_dataset.py](../data/aligned_dataset.py) includes a dataset class that can load aligned image pairs. It assumes a single image directory `/path/to/data/train`, which contains image pairs in the form of {A,B}. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md#prepare-your-own-datasets-for-pix2pix) on how to prepare aligned datasets. During test time, you need to prepare a directory `/path/to/data/test` for test data.
* [unaligned_dataset.py](../data/unaligned_dataset.py) includes a dataset class that can load unaligned/unpaired datasets. It assumes that two directories to host training images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB` separately. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Similarly, you need to prepare two directories `/path/to/data/testA` and `/path/to/data/testB` during test time.
* [single_dataset.py](../data/single_dataset.py) includes a dataset class that can load a set of single images. It is used in `test.py` when only model in one direction is being tested. The option `--model test` is used for generating CycleGAN results only for one side. This option will automatically set `--dataset_mode single`.
* [colorization_dataset.py](../data/colorization_dataset.py) implements a dataset class that can load a set of nature images in RGB, and convert RGB format into (L, ab) pairs. It is required by pix2pix-based colorization model (`--model colorization`).


[models](../models) directory contains core modules related to objective functions, optimizations, and network architectures.
* [\_\_init\_\_.py](../models/__init__.py)
* [base_model.py](../models/base_model.py)
* [template_model.py](../models/template_model.py)
* [pix2pix_model.py](../models/pix2pix_model.py)
* [colorization_model.py](../models/colorization_model.py)
* [cycle_gan_model.py](../models/cycle_gan_model.py)
* [networks.py](../models/networks.py) module implements network architectures (both generators and discriminators), as well as normalization layers, initialization, optimization scheduler (learning rate policy), and GAN loss function.
* [test_model.py](../models/test_model.py)

[options](../options) directory includes our option modules: training options, test options and basic options (used in both training and test).
* [\_\_init\_\_.py](../options/__init__.py) an empty file to make the `options` directory a package.
* [base_options.py](../options/base_options.py) includes options that are used in both training and test. It also implements a few helper functions such as parsing, printing, and saving the options. It also gathers additional options defined in `modify_commandline_options` functions in both dataset class and model class.
* [train_options.py](../options/train_options.py) includes options that are only used in training time.
* [test_options.py](../options/test_options.py) includes options that are only used in test time.


[util](../util) directory includes a misc collection of useful utility functions.
  * [\_\_init\_\_.py](../util/__init__.py): an empty file to make the `util` directory a package.
  * [get_data.py](../util/get_data.py)
  * [html.py](../util/html.py)
  * [image_pool.py](../util/image_pool.py)
  * [util.py](../util/util.py)
  * [visualizer.py](../util/visualizer.py)
