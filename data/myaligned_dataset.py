from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import tifffile as tiff
import numpy as np
from PIL import Image


class MyAlignedDataset(BaseDataset):
    """Custom aligned dataset class for TIFF images."""

    def __init__(self, opt):
        """Initialize the dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = opt.dataroot  # Assuming data is organized in pairs in the same directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int) -- a random integer for data indexing

        Returns:
            a dictionary containing A, B, A_paths, and B_paths
                A (tensor) -- an image in the input domain
                B (tensor) -- its corresponding image in the target domain
                A_paths (str) -- path to the input image
                B_paths (str) -- path to the target image
        """
        AB_path = self.AB_paths[index]
        AB = tiff.imread(AB_path)
        w, h = AB.shape[-1] // 2, AB.shape[-2] 
        A = Image.fromarray(AB[:, :w])
        B = Image.fromarray(AB[:, w:])
        
        # Convert tensor to numpy array and print all values
        A_array = A.numpy()
        print("All values of image A:", A_array)
        print("Shape:", A.size)
        print("Type:", A.mode)
        print("Min value:", np.min(A))
        print("Max value:", np.max(A))
        # Convert tensor to numpy array and print all values
        B_array = B.numpy()
        print("\nAll values of image B:", B_array)
        print("Shape:", B.size)
        print("Type:", B.mode)
        print("Min value:", np.min(B))
        print("Max value:", np.max(B))
        
        # apply the same transform to both A and B
        A = self.transform(A)
        B = self.transform(B)
        
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
