import torch.utils.data
import torchvision.transforms as transforms
from data.base_data_loader import BaseDataLoader
from data.image_folder import ImageFolder
# pip install future --upgrade
from builtins import object
from pdb import set_trace as st

class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        return self

    def __next__(self):
        A, A_paths = None, None
        B, B_paths = None, None
        try:
            A, A_paths = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A_paths is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A_paths = next(self.data_loader_A_iter)
        try:
            B, B_paths = next(self.data_loader_B_iter)

        except StopIteration:
            if B is None or B_paths is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B_paths = next(self.data_loader_B_iter)

        if self.stop_A and self.stop_B:
            self.stop_A = False
            self.stop_B = False
            raise StopIteration()
        else:
            return {'A': A, 'A_paths': A_paths,
                    'B': B, 'B_paths': B_paths}

class UnalignedDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        transform = transforms.Compose([
                                       transforms.Scale(opt.loadSize),
                                       transforms.CenterCrop(opt.fineSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))])

        # Dataset A
        dataset_A = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'A',
                                transform=transform, return_paths=True)
        data_loader_A = torch.utils.data.DataLoader(
            dataset_A,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))

        # Dataset B
        dataset_B = ImageFolder(root=opt.dataroot + '/' + opt.phase + 'B',
                                transform=transform, return_paths=True)
        data_loader_B = torch.utils.data.DataLoader(
            dataset_B,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches,
            num_workers=int(self.opt.nThreads))
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.paired_data = PairedData(data_loader_A, data_loader_B)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return max(len(self.dataset_A), len(self.dataset_B))
