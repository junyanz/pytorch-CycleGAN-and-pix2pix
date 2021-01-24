'''
Contains custom samplers for datasets
'''
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from torch.utils.data import Sampler as BaseSampler
from data.copd2class_dataset import NUM_PATCHES

class Copd2classSampler(BaseSampler):

    def __init__(self, dataset):
        self.size = len(dataset)   # NUM_PATCHES*set_size

    def __len__(self,):
        return self.size

    def __iter__(self):
        # Create an iterator
        numimgs = int(self.size/NUM_PATCHES)
        perm = np.random.permutation(numimgs)
        idx = []
        # For every index, create a random sequence within the patch
        for imgidx in perm:
            patchidx = np.random.permutation(NUM_PATCHES)
            for pid in patchidx:
                idx.append(NUM_PATCHES*imgidx + pid)

        return iter(idx)


if __name__ == "__main__":
    sampler = Copd2classSampler(np.arange(NUM_PATCHES*50))
    i = 0
    for s in sampler:
        print(s)
        i += 1
        if i >= 20:
            break
