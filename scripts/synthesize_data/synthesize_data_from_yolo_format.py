import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from models.networks import define_G
from util.utils import tensor2im
from datetime import datetime
from os.path import join
from pathlib import Path
import torch.nn as nn
from PIL import Image
import pandas as pd
import random
import torch
import cv2
import os


class RandomCrop(nn.Module):
    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        self.size = (size, size)

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), (j, i)


class ImageGenerator:
    def __init__(self, checkpoint_folder, epoch, n_crops, crop_size, input_nc, output_nc, tile=None):
        self.image_in_dirs = False
        self.epoch = epoch
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.n_crops = n_crops
        self.crop_size = crop_size
        self.tile = [tile, tile] if tile is not None else None

        load_filename = f'{epoch}_net_G_A.pth'
        self.load_path = join(checkpoint_folder, load_filename)
        self.generator_a = define_G(input_nc, output_nc, 64, 'resnet_9blocks', norm='instance')
        self.generator_a.load_state_dict(state_dict=torch.load(self.load_path, map_location=str(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))))
        self.generator_a.eval()

        self.greyscale = transforms.Grayscale(1)
        self.random_crop = RandomCrop(size=crop_size)
        # self.random_crop.valid_top_left = [(0,0)]
        self.to_tensor = transforms.ToTensor()
        self.greyscale_norm = transforms.Normalize((0.5,), (0.5,))
        self.rgb_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.now = datetime.now()

    def generate_images_from_folder(self):
        pass

    @staticmethod
    def _index_yolo_root(yolo_root, valid_scenes):
        all_data = []
        for zone in list(
                filter(lambda x: os.path.isdir(x), [os.path.join(yolo_root, x) for x in os.listdir(yolo_root)])):
            if zone == '/media/spidersense01/hdd/spectro/data/vis/train/yolo_format':
                continue
            zone_path = join(yolo_root, zone)
            for scene in os.listdir(zone_path):
                if (valid_scenes is not None) and (scene not in valid_scenes):
                    continue
                scene_images_path = join(zone_path, scene, 'images')
                scene_labels_path = join(zone_path, scene, 'labels')
                for image_name in os.listdir(scene_images_path):
                    base_image_name = image_name.split('.')[0]
                    all_data.append([join(scene_images_path, f'{base_image_name}.png'),
                                     join(scene_labels_path, f'{base_image_name}.txt')])
        return all_data

    def create_fake_b(self, fake_image, output_type):
        final_fake_b = []

        normalizer = self.greyscale_norm if self.input_nc == 1 else self.rgb_norm

        if output_type in ['tile']:
            if self.tile is None:
                raise Exception('The tile parameter is mandatory with tile outputs!')
            images = []
            for tile_row in range(self.crop_size // self.tile[0]):
                row_partial_images = []
                for tile_col in range(self.crop_size // self.tile[1]):
                    partial_image = fake_image.crop(((self.tile[1] * tile_col), (self.tile[0] * tile_row),
                                                     (self.tile[1] * (tile_col + 1)), (self.tile[0] * (tile_row + 1))))
                    partial_image = self.to_tensor(partial_image)
                    partial_image = normalizer(partial_image)

                    fake_b = self.generator_a(partial_image.unsqueeze(0))
                    row_partial_images.append(fake_b)
                images.append(torch.cat(row_partial_images, 3))
            fake_b = torch.cat(images, 2)
            final_fake_b.append(fake_b)
        if output_type in ['crop', 'both', 'all']:
            image = self.to_tensor(fake_image)
            image = normalizer(image)

            fake_b = self.generator_a(image.unsqueeze(0))
            final_fake_b.append(fake_b)
        if output_type not in ['crop', 'tile', 'both', 'all']:
            raise Exception('The output type is wrong!')
        return torch.cat(final_fake_b, 3)

    def generate_images_from_yolo_folder(self, yolo_root, output_type, valid_scenes=None,
                                         crops_with_objects_only=False):
        """
        :param crops_with_objects_only:
        :param valid_scenes: could be list of scenes or None (all valid)
        :param yolo_root: the folder with yolo format
        :param output_type:
            'crop' for generating image by crop size,
                'tile' for generating images by tile,
                'both' for concat both
                'all' for 'both' with original image
        :return: creates new yolo folder next to 'yolo_root'
        """
        all_data = self._index_yolo_root(yolo_root, valid_scenes)

        out_path = join(os.path.dirname(yolo_root),
                        f'yolo_format_{self.now.strftime("%d_%m_%Y__%H_%M_%S")}_generated_{self.n_crops}_crops_' +
                        f'with_size_{self.crop_size}' + (
                            f'_and_tile_{self.tile[0]}' if (self.tile is not None) and output_type in [
                                'tile', 'both', 'all'] else ''))
        os.makedirs(out_path)
        with open(join(out_path, 'meta_data.txt'), 'w') as meta_data:
            meta_data.write('\n'.join(
                [f'checkpoint: {self.load_path}', f'n_crops: {self.n_crops}', f'crop_size: {self.crop_size}']))

        n_crops = self.n_crops
        while n_crops > 0:
            if n_crops % 10 == 0:
                print(f'num crops left to generate: {n_crops}')
            data = random.choice(all_data)

            # verify the image not corrupted
            if Path(data[0]).stat().st_size < 1000:
                continue

            original_image = Image.open(data[0]).convert('RGB')
            image = original_image.copy()
            original_image_w, original_image_h = image.size

            if self.input_nc == 1:
                image = self.greyscale(image)
            image, (j, i) = self.random_crop(image)

            fake_b = self.create_fake_b(image, output_type=output_type)

            if output_type == 'all':
                cropped_original = original_image.crop((j, i, j + self.crop_size, i + self.crop_size))
                # bgr -> rgb
                b, g, r = cropped_original.split()
                cropped_original = Image.merge("RGB", (r, g, b))

                # apply to tensor and norm (because in the last step we apply un-norm and it should be with the same format as fake_b)
                cropped_original = self.to_tensor(cropped_original)
                cropped_original = self.rgb_norm(cropped_original)

                fake_b = torch.cat([cropped_original.unsqueeze(0), fake_b.repeat(1, 3 // self.output_nc, 1, 1)], 3)

            y_min, x_min = i, j
            image_labels = pd.read_csv(data[1], sep=' ', names=['class', 'cx', 'cy', 'w', 'h'])
            # translate and scale image
            image_labels['cx'] = image_labels['cx'] * original_image_w - x_min
            image_labels['w'] = image_labels['w'] * original_image_w
            image_labels['cy'] = image_labels['cy'] * original_image_h - y_min
            image_labels['h'] = image_labels['h'] * original_image_h

            # filter labels out of image
            filtered_image_labels = image_labels[(image_labels['cx'] > 0) & (image_labels['cy'] > 0) &
                                                 (image_labels['cx'] < self.crop_size) & (
                                                         image_labels['cy'] < self.crop_size)]

            # in case we want only crops with objects - check if the current crop is empty and skip accordingly
            if crops_with_objects_only and len(filtered_image_labels) == 0:
                continue

            filtered_image_labels['cx'] = filtered_image_labels['cx'] / self.crop_size
            filtered_image_labels['w'] = filtered_image_labels['w'] / self.crop_size
            filtered_image_labels['cy'] = filtered_image_labels['cy'] / self.crop_size
            filtered_image_labels['h'] = filtered_image_labels['h'] / self.crop_size

            image_parts = Path(data[0]).parts
            image_base_name = image_parts[-1].split('.')[0] + '0' + str(x_min) + '0' + str(y_min)
            new_images_path = join(out_path, '/'.join(image_parts[-4:-2]), 'images')
            os.makedirs(new_images_path, exist_ok=True)
            new_labels_path = join(out_path, '/'.join(image_parts[-4:-2]), 'labels')
            os.makedirs(new_labels_path, exist_ok=True)

            filtered_image_labels.reset_index(drop=True).round(5).to_csv(
                os.path.join(new_labels_path, image_base_name + '.txt'), index=False,
                header=False, sep=' ')
            cv2.imwrite(join(new_images_path, image_base_name + '.png'), tensor2im(fake_b))
            n_crops -= 1
        print('king')


def main(epoch, n_crops, crop_size, input_nc, output_nc, tile=None, crops_with_objects_only=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    pd.options.mode.chained_assignment = None

    yolo_root = 'C:/Users/Eliav/PycharmProjects/pytorch-CycleGAN-and-pix2pix/datasets/maps/yolo_folder'
    image_generator = ImageGenerator(
        checkpoint_folder='C:/Users/Eliav/PycharmProjects/pytorch-CycleGAN-and-pix2pix/checkpoints/experiment_name',
        epoch=epoch, n_crops=n_crops, crop_size=crop_size, input_nc=input_nc, output_nc=output_nc, tile=tile)

    image_generator.generate_images_from_yolo_folder(yolo_root, 'crop', crops_with_objects_only=crops_with_objects_only)


if __name__ == '__main__':
    # main(epoch=5, n_crops=1000, crop_size=10, input_nc=3, output_nc=3, crops_with_objects_only=True)
    main(epoch=5, n_crops=1000, crop_size=512, input_nc=3, output_nc=3, tile=256)
    main(epoch=5, n_crops=102, crop_size=512, input_nc=3, output_nc=3)
