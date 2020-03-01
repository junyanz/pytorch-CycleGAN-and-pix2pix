import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.true_labels = {}

        # Save true labels for each image pair
        with open('./datasets/mnist0123_labels.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                AB_path, label = line.replace('\n', '').split(' ')
                self.true_labels[AB_path] = label

        print('self.true_labels saved in AlignedDataset class')
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            true_label (str) - - true label of B image
        """
        # read a image given a random integer index
        print('---read image with random index ', index)
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        print('---AB ', AB)
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        print('---AB_path: %s, label: %s' % (AB_path, self.true_labels[AB_path]))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'true_label': self.true_labels[AB_path], 'labels_dict': self.true_labels}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
