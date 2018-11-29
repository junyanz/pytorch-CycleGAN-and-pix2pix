import numpy as np
import torch

from data.base_dataset import BaseDataset
from data.aligned2_dataset import Aligned2Dataset
from data.unaligned_dataset import UnalignedDataset


class Mixed2Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(serial_batches=True)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.unaligned_dataset = UnalignedDataset()
        self.unaligned_dataset.initialize(opt)
        self.unaligned_len = self.unaligned_dataset.__len__() // self.opt.batch_size * self.opt.batch_size
        self.unaligned_idx = 0
        self.aligned_dataset = Aligned2Dataset()
        self.aligned_dataset.initialize(opt)
        self.aligned_len = self.aligned_dataset.__len__() // self.opt.batch_size * self.opt.batch_size
        self.aligned_idx = 0
        self.total_len = self.unaligned_len + self.aligned_len
        self.choices = np.random.choice([0, 1], size=self.total_len//self.opt.batch_size, p=[float(self.unaligned_len)/self.total_len, float(self.aligned_len)/self.total_len])
        print("Mixed2Dataset: %d aligned, %d unaligned" % (self.aligned_len, self.unaligned_len))

    def __getitem__(self, index):
        if index == 0:
            self.aligned_dataset.shuffle()
            self.unaligned_dataset.shuffle()

        assert(index < self.total_len)
        if self.choices[index//self.opt.batch_size]:
            assert(self.aligned_idx < self.aligned_len)
            ret = self.aligned_dataset.__getitem__(self.aligned_idx)
            ret['data_aligned'] = True
            self.aligned_idx += 1
            return ret
        else:
            assert(self.unaligned_idx < self.unaligned_len)
            ret = self.unaligned_dataset.__getitem__(self.unaligned_idx)
            ret['data_aligned'] = False
            self.unaligned_idx += 1
            return ret

    def __len__(self):
        return self.total_len

    def name(self):
        return 'Mixed2Dataset'
