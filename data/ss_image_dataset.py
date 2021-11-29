from os import listdir
from os.path import isfile, join, isdir
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import numpy as np

from enum import Flag, auto


class ZoomLevelFlag(Flag):
    X1 = auto()
    X2 = auto()
    X3 = auto()
    CleanSet = X1
    CleanishSet = X1 | X2
    MessySet = X1 | X2 | X3


class AugmentFlag(Flag):
    NoAugments = 0x0
    AllowFlipHorizontal = auto()
    AllowFlipVertical = auto()
    AllowFlipBoth = AllowFlipHorizontal | AllowFlipVertical


def save_png(f_name, img):
    try:
        p_img = Image.fromarray(np.uint8(img)).convert('RGB')
        p_img.save(f_name, format='png')
    except:
        pass


def open_image(f_name, augment=AugmentFlag.NoAugments):
    # im = cv2.imread(f_name)
    # if im is None:
    try:
        im = Image.open(f_name)
        if augment & AugmentFlag.AllowFlipHorizontal != 0:
            im = ImageOps.mirror(im)
        if augment & AugmentFlag.AllowFlipVertical != 0:
            im = ImageOps.flip(im)

        im = np.array(im.convert('RGB'))
        return im
    except:
        return None


class SSImageDataset(Dataset):
    def __init__(self, root_dirs, image_sets, consoles,
                 zoom_levels=ZoomLevelFlag.CleanSet,
                 augments_allowed=AugmentFlag.NoAugments,
                 train_pct=0.7,
                 train=True,
                 split_seed=0,
                 transform=None,
                 filetype='png', exclude_noisy_files=False):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        root_dirs = [rd.removesuffix('/') for rd in root_dirs]
        noisy_flag = '__noisy__'
        if isinstance(image_sets, str):
            image_sets = [image_sets]

        if isinstance(consoles, str):
            consoles = [consoles]

        zoom_dirs = []
        if zoom_levels & ZoomLevelFlag.X1 == ZoomLevelFlag.X1:
            zoom_dirs.append('1x')
        if zoom_levels & ZoomLevelFlag.X2 == ZoomLevelFlag.X2:
            zoom_dirs.append('2x')
        if zoom_levels & ZoomLevelFlag.X3 == ZoomLevelFlag.X3:
            zoom_dirs.append('2x_')

        img_files = []

        for root_dir in root_dirs:
            for image_set in image_sets:
                for console in consoles:
                    base_dir = f'{root_dir}/{image_set}/{console}/cropped'
                    if not isdir(base_dir):
                        continue
                    for zl in zoom_dirs:
                        img_dir = f'{base_dir}/{zl}'
                        if not isdir(img_dir):
                            continue
                        img_files.extend([(AugmentFlag.NoAugments, f'{img_dir}/{f}') for f in listdir(img_dir)
                                          if f.endswith(f'.{filetype}')  # make sure the filetype is correct
                                          and isfile(join(img_dir, f))  # make sure it's a file
                                          and (not exclude_noisy_files  # include all if noisy files not excluded
                                               or (exclude_noisy_files  # or only include
                                                   and noisy_flag not in f))])  # non-noisy ones.

        self.augments_allowed = augments_allowed

        augments = []
        if (self.augments_allowed & AugmentFlag.AllowFlipHorizontal) != 0:
            # horizontal flip augments
            augments.extend([(AugmentFlag.AllowFlipHorizontal, f) for (_, f) in img_files])
        if (self.augments_allowed & AugmentFlag.AllowFlipVertical) != 0:
            # vertical flip augments
            augments.extend([(AugmentFlag.AllowFlipVertical, f) for (_, f) in img_files])
        if (self.augments_allowed & AugmentFlag.AllowFlipBoth) != 0:
            # vertical AND horizontal flip augments
            augments.extend([(AugmentFlag.AllowFlipBoth, f) for (_, f) in img_files])
        img_files.extend(augments)

        # divide into test/train set
        self.split_seed = split_seed
        self.train = True
        self.train_pct = train_pct
        rng = np.random.default_rng(self.split_seed)
        train_size = int(len(img_files) * min(max(self.train_pct, 0.), 1.))
        rng.shuffle(img_files)
        if train:
            img_files = img_files[:train_size]
        else:
            img_files = img_files[train_size:]

        self._img_files = np.array(img_files)
        self.transform = transform

    def get_filenames(self):
        filenames = [f for (_, f) in self._img_files]
        return filenames

    def __getitem__(self, idx) -> T_co:
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # else:
        #     try:
        #         iter(idx)
        #     except TypeError:
        #         idx = [idx]
        augment, filename = self._img_files[idx]
        img = open_image(filename, augment)
        # if img is None:
        #     print()
        if img is not None and self.transform is not None:
            img = self.transform(img)
        return img

        pass

    def __len__(self):
        return len(self._img_files)


class ImageToTensorTransform:
    def __call__(self, img):
        if img is not None:
            img = img.transpose((2, 0, 1))
            # print(img.shape)
            return torch.flatten(torch.from_numpy(img), 1)


class TensorToImageTransform:
    def __call__(self, t, height):
        if t is not None:
            img = t.numpy()
            img = img.reshape(img.shape[0], height, -1).transpose(1, 2, 0)
            return img
