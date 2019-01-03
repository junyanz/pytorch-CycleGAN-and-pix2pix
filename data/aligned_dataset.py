import os.path
import random
from data.base_dataset import BaseDataset, get_simple_transform
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')  # only support this mode
        assert(self.opt.load_size >= self.opt.crop_size)
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform_A = get_simple_transform(grayscale=(input_nc == 1))
        self.transform_B = get_simple_transform(grayscale=(output_nc == 1))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A0 = AB.crop((0, 0, w2, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        B0 = AB.crop((w2, 0, w, h)).resize((self.opt.load_size, self.opt.load_size), Image.BICUBIC)
        x, y, h, w = transforms.RandomCrop.get_params(A0, output_size=[self.opt.crop_size, self.opt.crop_size])
        A = TF.crop(A0, x, y, h, w)
        B = TF.crop(B0, x, y, h, w)

        if (not self.opt.no_flip) and random.random() < 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)
        A = self.transform_A(A)
        B = self.transform_B(B)
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)
