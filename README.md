<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# CycleGAN and pix2pix in PyTorch

This is our PyTorch implementation for both unpaired and paired image-to-image translation. It is still under active development.

The code was written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung89), and supported by [Tongzhou Wang](https://ssnl.github.io/).

This PyTorch implementation produces results comparable to or better than our original Torch software. If you would like to reproduce the exact same results as in the papers, check out the original [CycleGAN Torch](https://github.com/junyanz/CycleGAN) and [pix2pix Torch](https://github.com/phillipi/pix2pix) code

**Note**: The current software works well with PyTorch 0.4. Check out the older [branch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/pytorch0.3.1) that supports PyTorch 0.1-0.3.

**CycleGAN: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN)**
<img src="https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg" width="800"/>


**Pix2pix:  [Project](https://phillipi.github.io/pix2pix/) |  [Paper](https://arxiv.org/pdf/1611.07004v1.pdf) |  [Torch](https://github.com/phillipi/pix2pix)**

<img src="https://phillipi.github.io/pix2pix/images/teaser_v3.png" width="800px"/>


**[EdgesCats Demo](https://affinelayer.com/pixsrv/) | [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow) | by [Christopher Hesse](https://twitter.com/christophrhesse)**

<img src='imgs/edges2cats.jpg' width="400px"/>

If you use this code for your research, please cite:

Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz/)\*,  [Taesung Park](https://taesung.me/)\*, [Phillip Isola](https://people.eecs.berkeley.edu/~isola/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)
In ICCV 2017. (* equal contributions) [[Bibtex]](https://junyanz.github.io/CycleGAN/CycleGAN.txt)


Image-to-Image Translation with Conditional Adversarial Networks
[Phillip Isola](https://people.eecs.berkeley.edu/~isola), [Jun-Yan Zhu](https://people.eecs.berkeley.edu/~junyanz), [Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros)
In CVPR 2017. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

## Course
CycleGAN course assignment [code](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip) and [handout](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) designed by Prof. [Roger Grosse](http://www.cs.toronto.edu/~rgrosse/) for [CSC321](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/) "Intro to Neural Networks and Machine Learning" at University of Toronto. Please contact the instructor if you would like to adopt it in your course.

## Other implementations
### CycleGAN
<p><a href="https://github.com/leehomyc/cyclegan-1"> [Tensorflow]</a> (by Harry Yang),
<a href="https://github.com/architrathore/CycleGAN/">[Tensorflow]</a> (by Archit Rathore),
<a href="https://github.com/vanhuyz/CycleGAN-TensorFlow">[Tensorflow]</a> (by Van Huy),
<a href="https://github.com/XHUJOY/CycleGAN-tensorflow">[Tensorflow]</a> (by Xiaowei Hu),
<a href="https://github.com/LynnHo/CycleGAN-Tensorflow-Simple"> [Tensorflow-simple]</a> (by Zhenliang He),
<a href="https://github.com/luoxier/CycleGAN_Tensorlayer"> [TensorLayer]</a> (by luoxier),
<a href="https://github.com/Aixile/chainer-cyclegan">[Chainer]</a> (by Yanghua Jin),
<a href="https://github.com/yunjey/mnist-svhn-transfer">[Minimal PyTorch]</a> (by yunjey),
<a href="https://github.com/Ldpe2G/DeepLearningForFun/tree/master/Mxnet-Scala/CycleGAN">[Mxnet]</a> (by Ldpe2G),
<a href="https://github.com/tjwei/GANotebooks">[lasagne/keras]</a> (by tjwei)</p>
</ul>

### pix2pix
<p><a href="https://github.com/affinelayer/pix2pix-tensorflow"> [Tensorflow]</a> (by Christopher Hesse),
<a href="https://github.com/Eyyub/tensorflow-pix2pix">[Tensorflow]</a> (by Eyyüb Sariu),
<a href="https://github.com/datitran/face2face-demo"> [Tensorflow (face2face)]</a> (by Dat Tran),
<a href="https://github.com/awjuliani/Pix2Pix-Film"> [Tensorflow (film)]</a> (by Arthur Juliani),
<a href="https://github.com/kaonashi-tyc/zi2zi">[Tensorflow (zi2zi)]</a> (by Yuchen Tian),
<a href="https://github.com/pfnet-research/chainer-pix2pix">[Chainer]</a> (by mattya),
<a href="https://github.com/tjwei/GANotebooks">[tf/torch/keras/lasagne]</a> (by tjwei),
<a href="https://github.com/taey16/pix2pixBEGAN.pytorch">[Pytorch]</a> (by taey16)
</p>
</ul>

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch 0.4, torchvision, and other dependencies from http://pytorch.org
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom dominate
```
- Alternatively, all dependencies can be installed by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```
- For Conda users, we include a script `./scripts/conda_deps.sh` to install PyTorch and other libraries.

### CycleGAN train/test
- Download a CycleGAN dataset (e.g. maps):
```bash
bash ./datasets/download_cyclegan_dataset.sh maps
```
- Train a model:
```bash
#!./scripts/train_cyclegan.sh
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out `./checkpoints/maps_cyclegan/web/index.html`
- Test the model:
```bash
#!./scripts/test_cyclegan.sh
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```
The test results will be saved to a html file here: `./results/maps_cyclegan/latest_test/index.html`.

### pix2pix train/test
- Download a pix2pix dataset (e.g.facades):
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Train a model:
```bash
#!./scripts/train_pix2pix.sh
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_direction BtoA
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. To see more intermediate results, check out  `./checkpoints/facades_pix2pix/web/index.html`
- Test the model (`bash ./scripts/test_pix2pix.sh`):
```bash
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --which_direction BtoA
```
The test results will be saved to a html file here: `./results/facades_pix2pix/test_latest/index.html`.

More example scripts can be found at `scripts` directory.

### Apply a pre-trained model (CycleGAN)
- You can download a pretrained model (e.g. horse2zebra) with the following script:
```bash
bash ./scripts/download_cyclegan_model.sh horse2zebra
```
The pretrained model is saved at `./checkpoints/{name}_pretrained/latest_net_G.pth`. The available models are apple2orange, orange2apple, summer2winter_yosemite, winter2summer_yosemite, horse2zebra, zebra2horse, monet2photo, style_monet, style_cezanne, style_ukiyoe, style_vangogh, sat2map, map2sat, cityscapes_photo2label, cityscapes_label2photo, facades_photo2label, facades_label2photo, and iphone2dslr_flower.
- To test the model, you also need to download the  horse2zebra dataset:
```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

- Then generate the results using
```bash
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test
```
The option `--model test` is used for generating results of CycleGAN only for one side. `python test.py --model cycle_gan` will require loading and generating results in both directions, which is sometimes unnecessary. The results will be saved at `./results/`. Use `--results_dir {directory_path_to_save_result}` to specify the results directory.

- If you would like to apply a pre-trained model to a collection of input images (rather than image pairs), please use `--dataset_mode single` and `--model test` options. Here is a script to apply a model to Facade label maps (stored in the directory `facades/testB`).
``` bash
#!./scripts/test_single.sh
python test.py --dataroot ./datasets/facades/testB/ --name {your_trained_model_name} --model test
```
You might want to specify `--which_model_netG` to match the generator architecture of the trained model.

### Apply a pre-trained model (pix2pix)

Download a pre-trained model with `./scripts/download_pix2pix_model.sh`.

- For example, if you would like to download label2photo model on the Facades dataset,
```bash
bash ./scripts/download_pix2pix_model.sh facades_label2photo
```

- Download the pix2pix facades datasets
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
```
- Then generate the results using
```bash
python test.py --dataroot ./datasets/facades/ --which_direction BtoA --model pix2pix --name facades_label2photo_pretrained
```
Note that we specified `--which_direction BtoA` as Facades dataset's A to B direction is photos to labels.

- See a list of currently available models at `./scripts/download_pix2pix_model.sh`

## [Datasets](docs/datasets.md)
Download pix2pix/CycleGAN datasets and create your own datasets.

## [Training/Test Tips](docs/tips.md)
Best practice for training and testing your models.

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```



## Related Projects
**[CycleGAN-Torch](https://github.com/junyanz/CycleGAN) |
[pix2pix-Torch](https://github.com/phillipi/pix2pix) | [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) |
[iGAN](https://github.com/junyanz/iGAN) |
[BicycleGAN](https://github.com/junyanz/BicycleGAN)**

## Cat Paper Collection
If you love cats, and love reading cool graphics, vision, and learning papers, please check out the Cat Paper Collection:
[Github](https://github.com/junyanz/CatPapers) |  [Webpage](https://people.eecs.berkeley.edu/~junyanz/cat/cat_papers.html)

## Acknowledgments
Code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan).
