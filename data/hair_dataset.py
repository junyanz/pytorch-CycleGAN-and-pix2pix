

from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import os.path
import json
import numpy as np

class HairDataset(BaseDataset):
    """
    This dataset class loads the images for the haircolor_gan model.
    
    It requires that the files datasets/haircolor/hair_list_A.json and 
    datasets/haircolor/hair_list_B.json are present which each contain a list of filenames and RGB values. 
    An element of one of these lists is of the form
    [ filename (string) , red (float, between 0 and 1), blue (float, between 0 and 1), 
    green (float, between 0 and 1), fraction_of_img_that_is_hair (float, between 0 and 1) ].
    The corresponding images from the celebAMaskHQ dataset need to be in directory datasets/haircolor/images.
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--num_colors_to_match', type=int, default=5, help='number of images and colors from which to create pairs with large distance')
        parser.add_argument('--num_mc_drawings', type=int, default=100, help='number of drawings for the monte carlo algorithm that matches colors in' +
        'order to create pairs with large distance')
        parser.set_defaults(n_epochs=14)
        parser.set_defaults(n_epochs_decay = 12)
        parser.set_defaults(save_epoch_freq = 1)
        parser.set_defaults(save_latest_freq = 4000)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        A_data_list_path = os.path.join(opt.dataroot, 'hair_list_A.json')
        B_data_list_path = os.path.join(opt.dataroot, 'hair_list_B.json')
        
        A_data_list_file = open(A_data_list_path)
        self.A_data_list = json.load(A_data_list_file)
        A_data_list_file.close()
        
        B_data_list_file = open(B_data_list_path)
        self.B_data_list = json.load(B_data_list_file)
        B_data_list_file.close()        

        if opt.phase == 'train':
            self.A_data_list = self.A_data_list[:12000]
            self.A_data_list = self.B_data_list[:12000]
        elif opt.phase == 'test':
            self.A_data_list = self.A_data_list[12000:]
            self.A_data_list = self.B_data_list[12000:]
        else: 
            raise Exception("opt.phase is neither train nor test. Should be one of them.")

        self.A_size = len(self.A_data_list)
        self.B_size = len(self.B_data_list)
        
        self.dir_images = os.path.join(opt.dataroot, 'images')
        
        # define the transform function
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- an image in the target domain
            orig_color_A_img (tensor)    -- a monocolored image that shows the hair color of image A
            orig_color_B_img (tensor)    -- a monocolored image that shows the hair color of image B
            target_hair_color_img (tensor) -- a monocolored image that shows a random hair color from the target domain
            
            
        Rather than sampling one image image_A and one hair color target_hair_color directly from A_data_list and B_data_list, 
        we sample several (=opt.num_colors_to_match) images and and hair colors and match the different possiblities into pairs 
        The matching is optimized with a Monte Carlo algorithm so as to maximize the expected difference between original 
        hair color and target hair color. We then pick one of the pairs at random. 
        The algorithm is designed such that every hair color from train_B is equally likely to be picked, so the 
        distribution of hair colors is not changed as compared to the distribution in hair_B.
        similarly, every image from trainA is equally likely to be picked (if index is random).
        """

        index_B = random.randint(0,self.B_size - 1)
        B_list_entry = self.B_data_list[index_B]
        
        num_colors = self.opt.num_colors_to_match
        
        possible_A_list_entries = ( [self.A_data_list[index % self.A_size]] +
                                    [self.A_data_list[random.randint(0,self.A_size - 1)] for iii in range(num_colors-1)])
        possible_target_hair_colors = [self.B_data_list[random.randint(0,self.B_size - 1)] for iii in range(num_colors)]
        
        random.shuffle(possible_A_list_entries)
        A_list_entry = possible_A_list_entries[0]

        current_best_color_entry = possible_target_hair_colors[0]
        current_best_dist = color_distances(possible_A_list_entries, possible_target_hair_colors)
        for ii in range(self.opt.num_mc_drawings):
            random.shuffle(possible_target_hair_colors)
            current_dist = color_distances(possible_A_list_entries, possible_target_hair_colors)
            if current_dist > current_best_dist:
                current_best_dist = current_dist
                current_best_color_entry = possible_target_hair_colors[0]
                
        target_hair_color_entry = current_best_color_entry
        
        #get hair colors
        orig_color_A = A_list_entry[1:4]
        orig_color_B = B_list_entry[1:4]
        target_hair_color = target_hair_color_entry[1:4]
        orig_color_A_img = create_image_from_rgb(orig_color_A)
        orig_color_B_img = create_image_from_rgb(orig_color_B)
        target_hair_color_img = create_image_from_rgb(target_hair_color)
        
        #open image A and image B
        img_A_path = os.path.join(self.dir_images, A_list_entry[0])
        A_img = Image.open(img_A_path).convert('RGB')
        img_B_path = os.path.join(self.dir_images, B_list_entry[0])
        B_img = Image.open(img_B_path).convert('RGB')
        
        #apply image transformations
        A = self.transform(A_img)
        B = self.transform(B_img)
        orig_color_A_img = self.transform(orig_color_A_img)
        orig_color_B_img = self.transform(orig_color_B_img)
        target_hair_color_img = self.transform(target_hair_color_img)
        
        return {'A': A, 'B': B, 'orig_color_A_img': orig_color_A_img, 'orig_color_B_img': orig_color_B_img, 'target_hair_color_img':target_hair_color_img}

    def __len__(self):
        """Return the total number of images in input domain."""
        return self.A_size #TODO perhaps change this to take B_size into account.
        
def create_image_from_rgb(color, height = 256, width = 256):
    """Return an image where every pixel has the specified color
    
    Parameters:
        colors -- an array of length 3 containing floats which are intensities for the colors red,blue and green between 0 and 1
        height -- the height of the returned image in pixels
        width -- the width of the returned image in pixels
    Returns:
        an image with the specified height and width where every pixel has the specified color
    
    
    """
    colors_array = np.array(color)
    colors_array = np.reshape(colors_array,(1,1,3))
    
    img_array = np.ones((height,width,3)) * colors_array
    
    img = Image.fromarray(np.uint8(colors_array * 255))
    
    return img
        
def color_distances(list1,list2):
    """compute the sum of L2 distances between RGB values taken from list1 and list2
    
    Parameters:
        list1 -- a list where each element is a list that contains RGB values in positions with indices 1,2,3
        list2 -- a list (of same length as list1) where each element is a list that contains RGB values in positions with indices 1,2,3
    Returns:
        compute the sum of L2 distances between RGB values taken from list1 and list2
    """
    sum = 0.0
    for entry1, entry2 in zip(list1,list2):
        sum += np.linalg.norm(np.array(entry1[1:4]) - np.array(entry2[1:4]),ord=2)
    return sum