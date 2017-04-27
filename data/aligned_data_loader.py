import random
import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
from pdb import set_trace as st
# pip install future --upgrade
from builtins import object

class PairedData(object):
    def __init__(self, data_loader, fineSize, max_dataset_size):
        self.data_loader = data_loader
        self.fineSize = fineSize
        self.max_dataset_size = max_dataset_size
        # st()

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.iter = 0
        return self

    def __next__(self):
        self.iter += 1
        if self.iter > self.max_dataset_size:
            raise StopIteration

        AB, AB_paths = next(self.data_loader_iter)
        w_total = AB.size(3)
        w = int(w_total / 2)
        h = AB.size(2)

        w_offset = random.randint(0, max(0, w - self.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.fineSize - 1))
        A = AB[:, :, h_offset:h_offset + self.fineSize,
               w_offset:w_offset + self.fineSize]
        B = AB[:, :, h_offset:h_offset + self.fineSize,
               w + w_offset:w + w_offset + self.fineSize]

        return {'A': A, 'A_paths': AB_paths, 'B': B, 'B_paths': AB_paths}


class AlignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.fineSize = opt.fineSize
        transform = transforms.Compose([
            # TODO: Scale
            transforms.Scale(opt.loadSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        # Dataset A
        dataset = ImageFolder(root=opt.dataroot + '/' + opt.phase,
                              transform=transform, return_paths=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        self.dataset = dataset
        self.paired_data = PairedData(data_loader, opt.fineSize, opt.max_dataset_size)

    def name(self):
        return 'AlignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
