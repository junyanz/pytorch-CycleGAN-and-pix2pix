
# haircolorGAN

We build on the pyTorch implementation of pix2pix and cycleGAN at [pytorch-cycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) in order to implement a variation of the cycleGAN architecture for changing between multiple hair colors. The modified cycleGAN architecture is sketched in the following diagram. 


<img src="imgs/multiple_colors_architecture.png" width="800px"/>

Specifically I have added the following files:

- [haircolor_gan_model.py](models/haircolor_gan_model.py) - - implements the haircolorGAN model
- [hair_dataset.py](data/hair_dataset.py) - - data loader to load images and target hair colors for training
- [hair_testmode_dataset.py](data/hair_testmode.py) - - data loader for testing on specific (image,target color) combinations

I have also made modifications to the following files:

- [networks.py](models/networks.py) - - added the "SigmoidDiscriminator" discriminator architecture
- [train.py](train.py) - - made minor change in order to allow saving images multiple times per epoch during training.
- [visualizer.py](util/visualizer.py) - - disabled auto refresh for the html reports generated during training.

<img src='imgs/haircolorGAN_actress.png' width=600>

### Preparing the data

todo: write this section

### Training

todo: write this section

### Testing / Using trained model to translate hair colors

todo: write this section
