## Frequently Asked Questions
Before you post a new question, please first search for your questions in the following Q & A and existing GitHub issues. Many questions have been well addressed by us and other users with detailed answers. You may also want to read [Training/Test tips](docs/tips.md) for more suggestions.

#### Connection Error:HTTPConnectionPool (#230, #24, #38, etc.)
Similar error messages could be “Failed to establish a new connection/Connection refused”.  
Please start the visdom server before starting the training:
```bash
python -m visdom.server
```
To install the visdom, you can type the following command:
```bash
pip install visdom
```
You can also disable the visdom by setting `--display_id 0`.


####  “TypeError: Object of type 'Tensor' is not JSON serializable” (#258)
Similar errors: AttributeError: module 'torch' has no attribute 'device' (#314)

The current code only works well with PyTorch 0.4+. These errors are typically caused by an earlier PyTorch version.

#### Can I continue/resume my training? (#350, #275, #234, #87)
You can use the option `--continue_train`. By default, the program will initialize the epoch count as 1. Set `--epoch_count` to specify a different starting epoch count. See more discussion in training/test tips.

#### Why does my training loss not converge? (#335, #164, #30)
Many GAN losses do not converge (exception: WGAN, WGAN-GP) due to the nature of minimax optimization. For LSGAN loss, it’s quite normal if the G and D’s losses go up and down. It should be fine as long as they don’t blow up.

#### How can I make it work for my own data (e.g., 16-bit png, tiff, hyperspectral images)? (#309,  #320, #202)
The current code only supports RGB and grayscale images. If you would like to train the model on other data types, you need to do the following steps:

- change the parameters `--input_nc` and `--output_nc` to the number of channels in your input/output images.
- Write your own custom data loader (It is easy as long as you know how to load your data with python). If you write a new one, you need to change the flag `--dataset_mode` accordingly. Alternatively, you can modify the existing data loader. For aligned dataset, change this [line](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/aligned_dataset.py#L24); For unaligned dataset, change these two [lines](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/unaligned_dataset.py#L36).

- If you use visdom and/or HTML to visualize the results, you may also need to the visualization code.

#### Multi-GPU Training ( #327, #292, #137, #35)
You can use Multi-GPU training by setting `--gpu_ids` (e.g., `--gpu_ids 0,1,2,3` for the first four GPUs on your machine.) To fully utilize all the GPUs, you need to increase your batch size. Try `--batchSize 4`, `--batchSize 16`, or even a larger batchSize. Each GPU will process batchSize/#GPUs. The optimal batch size depends on the number of GPUs you have, GPU memory per GPU, and the resolution of training images.

We also recommend that you use instance normalization for multi-GPU training by setting `--norm instance`. The current batch normalization might not work for multi-GPUs as the batchnorm parameters are not shared across different GPUs. Advanced users can try synchronized [batchnorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch).


#### Can I run the model on CPU? (#310)
Yes, you can set `--gpu_ids 0`. See [training/test tips](docs/tips.md) for more details.


#### Are pretrained models available (#10)?
Yes, you can download pretrained models with the bash script `./scripts/download_cyclegan_model.sh`. See [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#apply-a-pre-trained-model-cyclegan) for more details. We are slowly adding more models to this repo.

#### Out of memory (#174)
CycleGAN is more memory-intensive than pix2pix as it requires two generators and two discriminators. If you would like to produce high-resolution images, you can do the following. First, train CycleGAN on cropped images of the training set. Please be careful not to change the aspect ratio or the scale of the original image, as this can lead to the training/test gap. You can usually do this by using `--resize_or_crop crop` option, or `--resize_or_crop scale_width_and_crop`. Then at test time, you can load only one generator to produce the results in a single direction. This greatly saves GPU memory as you are not loading the discriminators and the other generator in the opposite direction. You can probably take the whole image as input (we have done image generation of 1024x512 resolution). You can do this using `--model test --dataroot [path to the directory that contains your test images (e.g., ./datasets/horse2zebra/trainA)] --model_suffix _A`. For more explanation, please see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/test_model.py#L16.


#### The color gets inverted from the beginning of training (#249)
The authors also observe that the generator unnecessarily inverts the color of the input image early in training, and then never learns to undo the inversion. In this case, you can try  two things. First, try using identity loss `--identity 1.0` or `--identity 0.1`. We observe that the identity loss makes the generator to be more conservative and make fewer unnecessary changes. However, because of this, the change may not be as dramatic. Second, try smaller variance when initializing weights by changing `--init_gain`. We observe that smaller variance in weight initialization results in less color inversion.

#### For labels2phot Cityscapes evaluation, why does the pretrained FCN-8s model not work well on the original Cityscapes input images? (#150)
The model was trained on 256x256 images that are resized/upsampled to 1024x2048, so expected input images to the network are very blurry. The purpose of the resizing was to 1) keep the label maps in the original high resolution untouched and 2) avoid the need of changing the standard FCN training code for Cityscapes.

#### How do I get the `ground-truth` numbers on the labels2photo Cityscapes evaluation? (#150)
You need to resize the original Cityscapes images to 256x256 before running the evaluation code.



#### Using resize-conv to reduce checkerboard artifacts (#190, #64)?
This Distill [blog](https://distill.pub/2016/deconv-checkerboard/) discussed one of the potential causes of the checkerboard artifacts. You can fix that issue by switching from "deconvolution" to nearest-neighbor upsampling followed by regular convolution. Here is one implementation provided by @SSNL. You can replace the ConvTranspose2d with the following layers.
```python
nn.Upsample(scale_factor = 2, mode='bilinear'),
nn.ReflectionPad2d(1),
nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0),
```
We have also noticed that sometimes the checkboard artifacts will go away if you simply train long enough. Maybe you can try training your model a bit longer.

#### pix2pix/CycleGAN has no random noise z (#152)?
The current pix2pix/CycleGAN model does not take z as input. In both pix2pix and CycleGAN, we tried to add z to the generator: e.g., adding z to a latent state, concatenating with a latent state, applying dropout, etc., but often found the output did not vary significantly as a function of z. Conditional GANs do not need noise as long as the input is sufficiently complex so that the input can kind of play the role of noise. Without noise, the mapping is deterministic.

Please check out the following papers that show ways of getting z to actually have a substantial effect: e.g., [BicycleGAN](https://github.com/junyanz/BicycleGAN),  [AugmentedCycleGAN](https://arxiv.org/abs/1802.10151), [MUNIT](https://arxiv.org/abs/1804.04732), [DRIT](https://arxiv.org/pdf/1808.00948.pdf), etc.

#### Experiment details (e.g., BW->color) ( #306)
You can find more training details and hyperparameter settings in the appendix of [CycleGAN](https://arxiv.org/abs/1703.10593) and [pix2pix](https://arxiv.org/abs/1611.07004) papers.
