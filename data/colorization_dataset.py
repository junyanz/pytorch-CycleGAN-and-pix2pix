import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from skimage import color  # require skimage
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


class ColorizationDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=1, output_nc=2, direction='AtoB')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot)
        self.A_paths = sorted(make_dataset(self.dir_A))
        assert(opt.input_nc == 1 and opt.output_nc == 2 and opt.direction == 'AtoB')
        self.transform = get_transform(opt, convert=False)

    def __getitem__(self, index):
        path = self.A_paths[index]
        im = Image.open(path).convert('RGB')
        im = self.transform(im)
        im = np.array(im)
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        A = lab_t[[0], ...] / 50.0 - 1.0
        B = lab_t[[1, 2], ...] / 110.0
        return {'A': A, 'B': B, 'A_paths': path, 'B_paths': path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'ColorizationDataset'
