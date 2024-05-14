from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import tifffile as tiff
from PIL import Image

class MyDataset(BaseDataset):
    """Custom dataset class."""

     def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = tiff.imread(A_path)  # Load the TIFF image
        print("Valori dell'immagine prima del dataloader:")
        print(A_img)
        A_img = Image.fromarray(A_img.squeeze(), mode='L')  # Convert the NumPy array to a PIL image
        print("Valori dell'immagine dopo prime operazioni:")
        print(A_img)
        A = self.transform(A_img)
        print("Valori dell'immagine dopo transform:")
        print(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
