import random
import numpy as np
from PIL import Image, ImageOps

from data.base_dataset import BaseDataset, get_base_transform_list
from data.ss_image_dataset import SSImageDataset, AugmentFlag, ZoomLevelFlag


class TriforceDataset(BaseDataset):
    """
    This dataset class can load the triforce datasets.

    It requires two sub-datasets, each with a different set of images. (E.g. 2 different consoles).
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        split = opt.tf_split
        self.console_a = opt.tf_console_a
        self.console_b = opt.tf_console_b

        dataset_maker = lambda console: SSImageDataset(root_dirs=self.root,
                                                       image_sets=['internetarchive',
                                                                   'mobygames',
                                                                   'superfamicomdotorg',
                                                                   ],
                                                       train=opt.isTrain,
                                                       train_pct=split,
                                                       consoles=[console],
                                                       augments_allowed=AugmentFlag.AllowFlipBoth,
                                                       zoom_levels=ZoomLevelFlag.CleanSet,
                                                       # transform=ImageToTensorTransform(),
                                                       exclude_noisy_files=False)
        self.dataset_a = dataset_maker(self.console_a)
        self.dataset_b = dataset_maker(self.console_b)

        max_data_size = min(min(len(self.dataset_a), len(self.dataset_b)), opt.max_dataset_size)
        self.A_paths = list(sorted([p for p in self.dataset_a.get_filenames()][:max_data_size]))
        self.B_paths = list(sorted([p for p in self.dataset_b.get_filenames()][:max_data_size]))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_triforce_transform(opt=self.opt,
                                                  console=self.console_a,
                                                  grayscale=(input_nc == 1),
                                                  method=Image.NEAREST)
        self.transform_B = get_triforce_transform(opt=self.opt,
                                                  console=self.console_b,
                                                  grayscale=(output_nc == 1),
                                                  method=Image.NEAREST)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


def get_triforce_transform(opt, console, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    import torchvision.transforms as transforms
    transform_list = get_base_transform_list(opt, params, grayscale, method, convert)

    # add color conversion transform
    if console == 'nes':
        transform_list.insert(0, TriforceQuantizeColorsToNESTransform())
    elif console == 'snes':
        transform_list.insert(0, TriforceQuantizeColorsToSNESTransform())

    # ensure image dims are divisible by 256 if necessary
    if 'unet_' in opt.netG:
        transform_list.insert(0, TriforcePaddingTransform(output_size=(256, 256)))
    return transforms.Compose(transform_list)


class TriforcePaddingTransform:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img):
        img_w, img_h = img.size
        out_w, out_h = self.output_size

        d_w = out_w - img_w
        d_h = out_h - img_h
        padding = (d_w // 2, d_h // 2, d_w - (d_w // 2), d_h - (d_h // 2))
        return ImageOps.expand(img, padding)


class TriforceQuantizeColorsToSNESTransform:
    def __init__(self):
        pass

    def __call__(self, img):

        pixel_map = np.asarray(img)
        pixel_map = pixel_map & 0xF8
        return Image.fromarray(pixel_map)


class TriforceQuantizeColorsToNESTransform:
    def __init__(self):
        self.palette = sum([[r, g, b] for (r, g, b) in [(84, 84, 84),
                                                        (0, 30, 116),
                                                        (8, 16, 144),
                                                        (48, 0, 136),
                                                        (68, 0, 100),
                                                        (92, 0, 48),
                                                        (84, 4, 0),
                                                        (60, 24, 0),
                                                        (32, 42, 0),
                                                        (8, 58, 0),
                                                        (0, 64, 0),
                                                        (0, 60, 0),
                                                        (0, 50, 60),
                                                        (0, 0, 0),
                                                        (152, 150, 152),
                                                        (8, 76, 196),
                                                        (48, 50, 236),
                                                        (92, 30, 228),
                                                        (136, 20, 176),
                                                        (160, 20, 100),
                                                        (152, 34, 32),
                                                        (120, 60, 0),
                                                        (84, 90, 0),
                                                        (40, 114, 0),
                                                        (8, 124, 0),
                                                        (0, 118, 40),
                                                        (0, 102, 120),
                                                        (0, 0, 0),
                                                        (236, 238, 236),
                                                        (76, 154, 236),
                                                        (120, 124, 236),
                                                        (176, 98, 236),
                                                        (228, 84, 236),
                                                        (236, 88, 180),
                                                        (236, 106, 100),
                                                        (212, 136, 32),
                                                        (160, 170, 0),
                                                        (116, 196, 0),
                                                        (76, 208, 32),
                                                        (56, 204, 108),
                                                        (56, 180, 204),
                                                        (60, 60, 60),
                                                        (236, 238, 236),
                                                        (168, 204, 236),
                                                        (188, 188, 236),
                                                        (212, 178, 236),
                                                        (236, 174, 236),
                                                        (236, 174, 212),
                                                        (236, 180, 176),
                                                        (228, 196, 144),
                                                        (204, 210, 120),
                                                        (180, 222, 120),
                                                        (168, 226, 144),
                                                        (152, 226, 180),
                                                        (160, 214, 228),
                                                        (160, 162, 160)]], [])

        """Convert an RGB or L mode image to use a given P image's palette."""
        self.pal_image = Image.new('P', (16, 16))
        self.pal_image.putpalette(self.palette)
        self.pal_image.load()

    def __call__(self, img):
        img.load()
        im = img.im.convert("P", 0, self.pal_image.im)
        # the 0 above means turn OFF dithering making solid colors
        return img._new(im).convert('RGB')
        pass
