import glob
import os
from options.test_options import TestOptions
from data import create_dataset, base_dataset
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im
from skimage.morphology import convex_hull_image
from PIL import Image
import torch
import numpy as np
import cv2
import dlib
import torchvision.transforms as transforms


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch. device('cpu')
    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(opt.dlib_path)
    # do data preprocess here explicitly
    test_files = glob.glob(opt.dataroot+'/testA/*.jpg')
    for file in test_files:
        img = cv2.imread(file, -1)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        face_rect = detector(img, 1)
        if len(face_rect) == 0:
            continue
        kpt = predictor(img, face_rect[0])
        for i in range(68):
            x, y = kpt.part(i).x, kpt.part(i).y
            mask[min(h-1, max(y, 0)), min(w-1, max(0, x))] = 1
        mask = convex_hull_image(mask)
        x0, x1, y0, y1 = base_dataset.bounding_rect(mask)
        # additional step to make image squared
        pad_size = int(abs(x1+y0-x0-y1))
        print(x0, x1, y0, y1, y1-y0, x1-x0, img.shape)
        padding = (int(pad_size//2), 0, pad_size-int(pad_size//2), 0) if (y1-y0)>(x1-x0) else (0, int(pad_size//2), 0, pad_size-int(pad_size//2))
        img_tensor = Image.fromarray(img[y0: y1, x0: x1, ::-1]).convert('RGB')
        img_tensor.save('tmp_ori.jpg')
        print('forward shape:', img_tensor.size)
        #transform = transforms.Compose([transforms.Pad(padding), transforms.Resize((256, 256))])
        #img_tensor = transform(img_tensor)
        img_tensor = transforms.functional.pad(img_tensor, padding)
        print('\tpadded shape:', img_tensor.size, padding, pad_size)
        ori_size = img_tensor.size
        img_tensor = transforms.functional.resize(img_tensor, (256, 256), Image.BICUBIC)
        print('\tresize shape:', img_tensor.size)
        img_tensor = transforms.functional.to_tensor(img_tensor)
        img_tensor = transforms.functional.normalize(img_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)
        print('tensor shape:', img_tensor.shape)

        img_norm = model.netG_A(img_tensor)
        print('\ttransform shape:', img_norm.shape)
        img_norm = Image.fromarray(tensor2im(img_norm))
        print('output shape:', img_norm.shape)
        img_reverse = transforms.functional.resize(img_norm, ori_size)
        print('reverse resize shape:', img_reverse.shape)
        img_reverse = img_reverse[padding[1]: img_reverse.shape[0]-padding[3], padding[0]: img_reverse.shape[1]-padding[2], :]
        print('reverse shape:', img_reverse.shape, img_reverse.min(), img_reverse.max())
        img_reverse.save('tmp.jpg')
        break
