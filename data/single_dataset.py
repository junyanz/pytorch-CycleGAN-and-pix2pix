import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        transform_list = []
        if opt.resize_or_crop == 'resize_and_crop':
            transform_list.append(transforms.Scale(opt.loadSize))

        if opt.isTrain and not opt.no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        if opt.resize_or_crop != 'no_resize':
            transform_list.append(transforms.RandomCrop(opt.fineSize))

        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        A_img = Image.open(A_path).convert('RGB')

        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
