import unittest
from unittest.mock import patch
import torch

import sys
import argparse

from options.train_options import TrainOptions

class TestTrainOptions(unittest.TestCase):

    required_args = ['train.py', '--dataroot', '/tmp']


    def test_torch_devices_default(self):
        with patch('sys.argv', self.required_args + []):
            b = TrainOptions()
            opt = b.parse()
            self.assertEqual(opt.device_type, 'cuda')
            self.assertEqual(opt.torch_devices, [torch.device('cuda', 0)])

    def test_torch_devices_from_gpu_ids(self):
        with patch.object(sys, 'argv', self.required_args + ['--gpu_ids', '0']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cuda', 0)])

        with patch.object(sys, 'argv', self.required_args + ['--gpu_ids', '1']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cuda', 1)])

        with patch.object(sys, 'argv', self.required_args + ['--gpu_ids', '0,2']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cuda', 0), torch.device('cuda', 2)])

    def test_torch_devices_gpu_ids_type_conflict(self):
        with patch.object(sys, 'argv', self.required_args + ['--gpu_ids', '0', '--device_type', 'cpu']):
            with self.assertRaises(SystemExit) as cm:
               TrainOptions().parse()


    def test_torch_devices_mps(self):
        with patch('sys.argv', self.required_args + ['--device_type', 'mps']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('mps', 0)])

        with patch('sys.argv', self.required_args + ['--device_type', 'mps', '--device_ids', '0']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('mps', 0)])

        with patch.object(sys, 'argv', self.required_args + ['--device_type', 'mps', '--device_ids', '2']):
            with self.assertRaises(SystemExit) as cm:
               TrainOptions().parse()

    def test_torch_devices_cpu(self):
        with patch('sys.argv', self.required_args + ['--device_type', 'cpu']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cpu', 0)])

        with patch('sys.argv', self.required_args + ['--device_type', 'cpu', '--device_ids', '0,2']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cpu', 0), torch.device('cpu', 2)])



    def test_torch_devices_cuda(self):
        with patch('sys.argv', self.required_args + ['--device_type', 'cuda']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cuda', 0)])

        with patch('sys.argv', self.required_args + ['--device_type', 'cuda', '--device_ids', '0,2']):
            opt = TrainOptions().parse()
            self.assertEqual(opt.torch_devices, [torch.device('cuda', 0), torch.device('cuda', 2)])










