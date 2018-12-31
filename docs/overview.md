## Overview of Code Structure
* [data](./data) package contains all the modules related to datasets.
  * [aligned_dataset.py](./data/aligned_dataset.py)
  * [base_data_loader.py](./data/base_data_loader.py)
  * [base_dataset.py](./data/base_dataset.py)
  * [colorization_dataset.py](./data/colorization_dataset.py)
  * [image_folder.py](./data/image_folder.py)
  * [\__init__.py](./data/__init__.py)
  * [single_dataset.py](./data/single_dataset.py)
  * [template_dataset.py](./data/template_dataset.py)
  * [unaligned_dataset.py](./data/unaligned_dataset.py)
* [models](./models) package contains all the modules related to core core formulations and network.
  * [base_model.py](./models/base_model.py)
  * [cycle_gan_model.py](./models/cycle_gan_model.py)
  * [\__init__.py](./models/__init__.py)
  * [networks.py](./models/networks.py) module implements network architectures (both generators and discriminators), as well as normalization layers, initialization, optimization scheduler (learning rate policy), and GAN loss function.
  * [colorization_model.py](./models/colorization_model.py)
  * [pix2pix_model.py](./models/pix2pix_model.py)
  * [template_model.py](./models/template_model.py)
  * [test_model.py](./models/test_model.py)
* [options](./options) package includes option modules: training options, test options and basic options (used in both training and test).
  * [base_options.py](./options/base_options.py)
  * [\__init__.py](./options/__init__.py)
  * [test_options.py](./options/test_options.py)
  * [train_options.py](./options/train_options.py)
* [test.py](./test.py) script: a general-purpose training script.
* [train.py](./train.py) script: a general-purpose test script.
* [util](./util) package includes a misc collection of useful utility functions.
    * [get_data.py](./util/get_data.py)
    * [html.py](./util/html.py)
    * [image_pool.py](./util/image_pool.py)
    * [\__init__.py](./util/__init__.py)
    * [util.py](./util/util.py)
    * [visualizer.py](./util/visualizer.py)
