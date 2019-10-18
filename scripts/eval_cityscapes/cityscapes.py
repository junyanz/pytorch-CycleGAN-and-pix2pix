# The following code is modified from https://github.com/shelhamer/clockwork-fcn
import sys
import os
import glob
import numpy as np
from PIL import Image


class cityscapes:
    def __init__(self, data_path):
        # data_path something like /data2/cityscapes
        self.dir = data_path
        self.classes = ['road', 'sidewalk', 'building', 'wall', 'fence',
                        'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                        'sky', 'person', 'rider', 'car', 'truck',
                        'bus', 'train', 'motorcycle', 'bicycle']
        self.mean = np.array((72.78044, 83.21195, 73.45286), dtype=np.float32)
        # import cityscapes label helper and set up label mappings
        sys.path.insert(0, '{}/scripts/helpers/'.format(self.dir))
        labels = __import__('labels')
        self.id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
        self.trainId2color = {label.trainId: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples

    def get_dset(self, split):
        '''
        List images as (city, id) for the specified split

        TODO(shelhamer) generate splits from cityscapes itself, instead of
        relying on these separately made text files.
        '''
        if split == 'train':
            dataset = open('{}/ImageSets/segFine/train.txt'.format(self.dir)).read().splitlines()
        else:
            dataset = open('{}/ImageSets/segFine/val.txt'.format(self.dir)).read().splitlines()
        return [(item.split('/')[0], item.split('/')[1]) for item in dataset]

    def load_image(self, split, city, idx):
        im = Image.open('{}/leftImg8bit_sequence/{}/{}/{}_leftImg8bit.png'.format(self.dir, split, city, idx))
        return im

    def assign_trainIds(self, label):
        """
        Map the given label IDs to the train IDs appropriate for training
        Use the label mapping provided in labels.py from the cityscapes scripts
        """
        label = np.array(label, dtype=np.float32)
        if sys.version_info[0] < 3:
            for k, v in self.id2trainId.iteritems():
                label[label == k] = v
        else:
            for k, v in self.id2trainId.items():
                label[label == k] = v
        return label

    def load_label(self, split, city, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = Image.open('{}/gtFine/{}/{}/{}_gtFine_labelIds.png'.format(self.dir, split, city, idx))
        label = self.assign_trainIds(label)  # get proper labels for eval
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

    def preprocess(self, im):
        """
        Preprocess loaded image (by load_image) for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def palette(self, label):
        '''
        Map trainIds to colors as specified in labels.py
        '''
        if label.ndim == 3:
            label = label[0]
        color = np.empty((label.shape[0], label.shape[1], 3))
        if sys.version_info[0] < 3:
            for k, v in self.trainId2color.iteritems():
                color[label == k, :] = v
        else:
            for k, v in self.trainId2color.items():
                color[label == k, :] = v
        return color

    def make_boundaries(label, thickness=None):
        """
        Input is an image label, output is a numpy array mask encoding the boundaries of the objects
        Extract pixels at the true boundary by dilation - erosion of label.
        Don't just pick the void label as it is not exclusive to the boundaries.
        """
        assert(thickness is not None)
        import skimage.morphology as skm
        void = 255
        mask = np.logical_and(label > 0, label != void)[0]
        selem = skm.disk(thickness)
        boundaries = np.logical_xor(skm.dilation(mask, selem),
                                    skm.erosion(mask, selem))
        return boundaries

    def list_label_frames(self, split):
        """
        Select labeled frames from a split for evaluation
        collected as (city, shot, idx) tuples
        """
        def file2idx(f):
            """Helper to convert file path into frame ID"""
            city, shot, frame = (os.path.basename(f).split('_')[:3])
            return "_".join([city, shot, frame])
        frames = []
        cities = [os.path.basename(f) for f in glob.glob('{}/gtFine/{}/*'.format(self.dir, split))]
        for c in cities:
            files = sorted(glob.glob('{}/gtFine/{}/{}/*labelIds.png'.format(self.dir, split, c)))
            frames.extend([file2idx(f) for f in files])
        return frames

    def collect_frame_sequence(self, split, idx, length):
        """
        Collect sequence of frames preceding (and including) a labeled frame
        as a list of Images.

        Note: 19 preceding frames are provided for each labeled frame.
        """
        SEQ_LEN = length
        city, shot, frame = idx.split('_')
        frame = int(frame)
        frame_seq = []
        for i in range(frame - SEQ_LEN, frame + 1):
            frame_path = '{0}/leftImg8bit_sequence/val/{1}/{1}_{2}_{3:0>6d}_leftImg8bit.png'.format(
                self.dir, city, shot, i)
            frame_seq.append(Image.open(frame_path))
        return frame_seq
