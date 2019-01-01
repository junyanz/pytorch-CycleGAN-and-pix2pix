## Overview of Code Structure
[train.py](./train.py) is a general-purpose training script. It works for various models (with model option `--model`: e.g., `pix2pix`, `cyclegan`, `colorization`) and different datasets (with dataset option `--dataset_mode`: e.g., `aligned`, `unaligned`, `single`, `colorization`). See the main [README](../README.md) and Training/test [tips](tips.md) for more details.

[test.py](./test.py) is a general-purpose test script. Once you have trained your models with `train.py`, you can use this script to test the model. It will load a saved model from `--checkpoints_dir` and save the results to `--results_dir`. See the main [README](../README.md) and Training/test [tips](tips.md) for more details.


[data](./data) directory contains all the modules related to datasets.
* [\_\_init\_\_.py](./data/__init__.py) implements the interface between this package and training/test script. You should call `from data import CreateDataLoader` and `data_loader = CreateDataLoader(opt)` to create a dataloader given an option `opt`.
* [base_dataset.py](./data/base_dataset.py) implements an abstract base dataset class. It also includes common transformation functions `get_transform` and `get_simple_transform` which can be used in dataset classes. To add a custom dataset class called `dummy`, you need to add a file called `dummy_dataset.py` and define a subclass `DummyDataset` inherited from `BaseDataset`. You need to implement functions: `name`, `__len__`, `__getitem__`, and `modify_commandline_options`. You can use this dataset using `--dataset_mode dummy`.
* [image_folder.py](./data/image_folder.py) implements a modified Image folder class. We
modify the original PyTorch code so that it also loads images from the current directory as well as the subdirectories.
* [template_dataset.py](./data/template_dataset.py) provides a class template with detailed documentation. Check out this file if you plan to implement your own dataset class.
* [aligned_dataset.py](./data/aligned_dataset.py) includes a dataset class that can load aligned image pairs.
* [unaligned_dataset.py](./data/unaligned_dataset.py) includes a dataset class that can load unaligned/unpaired datasets.
* [single_dataset.py](./data/single_dataset.py) includes a dataset class that can load a collection of single images. It is used in `test.py` when only model in one direction is being tested.
* [colorization_dataset.py](./data/colorization_dataset.py) implements a dataset class that can load a collection of RGB images and convert it into (L, ab) pairs. It is used with pix2pix-based colorization model.


[models](./models) directory contains all the modules related to core core formulations and network.
* [\_\_init\_\_.py](./models/__init__.py)
* [base_model.py](./models/base_model.py)
* [template_model.py](./models/template_model.py)
* [pix2pix_model.py](./models/pix2pix_model.py)
* [colorization_model.py](./models/colorization_model.py)
* [cycle_gan_model.py](./models/cycle_gan_model.py)
* [networks.py](./models/networks.py) module implements network architectures (both generators and discriminators), as well as normalization layers, initialization, optimization scheduler (learning rate policy), and GAN loss function.
* [test_model.py](./models/test_model.py)

[options](./options) directory includes option modules: training options, test options and basic options (used in both training and test).
* [\_\_init\_\_.py](./options/__init__.py) an empty file to make the `options` directory a package.
* [base_options.py](./options/base_options.py) includes options that are used in both training and test. It also implements a few helper functions such as parsing, printing, and saving the options. It also gathers additional options defined in `modify_commandline_options` functions in both dataset class and model class.
* [train_options.py](./options/train_options.py) includes options that are only used in training time.
* [test_options.py](./options/test_options.py) includes options that are only used in test time.


[util](./util) directory includes a misc collection of useful utility functions.
  * [\_\_init\_\_.py](./util/__init__.py): an empty file to make the `util` directory a package.
  * [get_data.py](./util/get_data.py)
  * [html.py](./util/html.py)
  * [image_pool.py](./util/image_pool.py)
  * [util.py](./util/util.py)
  * [visualizer.py](./util/visualizer.py)
