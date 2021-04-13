"""
Main changes:
1, allow to use ResNet
2, allow to only train slice-level network
"""

import os
import argparse
import builtins
import math
import random
import shutil
import time
import warnings
import json
import numpy as np
from tensorboard_logger import configure, log_value

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from moco.builder_MoCo_slice import MoCo as MoCo_Slice
import moco.loader
from monai.transforms import Compose, RandGaussianNoise, Rand2DElastic, RandAdjustContrast, RandAffine, Resize, RandFlip
from data.copd_MoCo_slice import COPD_dataset as COPD_dataset_slice

import models.cnn2d as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='2D CT Images MoCo Self-Supervised Training Slice-level')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--workers-slice', default=10, type=int, metavar='N',
                    help='slice-level number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size-slice', default=128, type=int,
                    metavar='N',
                    help='slice-level mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', default=[0.6, 0.8], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume-slice', default='', type=str, metavar='PATH',
                    help='path to latest slice-level checkpoint (default: None)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_false',
                    help='use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--npgus-per-node', default=2, type=int,
                    help='number of gpus per node.')

# COPD data configs:
parser.add_argument('--num-slice', default=379, type=int,
                    help='total number of slices in the atlas image.')
parser.add_argument('--root-dir', default=None,
                    help='root directory of registered images in COPDGene dataset')
parser.add_argument('--label-name', default=["FEV1pp_utah", "FEV1_FVC_utah", "finalGold"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--label-name-set2', default=["Exacerbation_Frequency", "MMRCDyspneaScor"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--visual-score', default=["Emph_Severity", "Emph_Paraseptal"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--P2-Pheno', default=["Exacerbation_Frequency_P2"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--fold', default=0, type=int,
                    help='fold index of cross validation')
parser.add_argument('--nhw-only', action='store_true',
                    help='only include white people')

# MoCo specific configs:
parser.add_argument('--rep-dim-slice', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--moco-dim-slice', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k-slice', default=512, type=int,
                    help='queue size; number of negative keys (default: 4096)')
parser.add_argument('--moco-m-slice', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t-slice', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')

# options for moco v2
parser.add_argument('--mlp-slice', action='store_false',
                    help='use mlp head')
parser.add_argument('--cos-slice', action='store_false',
                    help='use cosine lr schedule')

# experiment configs
parser.add_argument('--transform-type', default='affine', type=str,
                    help='image transformation type, affine or elastic (default: affine)')
parser.add_argument('--slice-size', default=224, type=int,
                    help='slice H, W, original size = 447 (default: 447)')
parser.add_argument('--mask-threshold', default=0.05, type=float,
                    help='lung mask threshold.')
parser.add_argument('--mask-imputation', action='store_true',
                    help='whether imputating region outside lung mask to -1024. default: no lung mask imputation')
parser.add_argument('--sample-prop', default=0.1, type=float,
                    help='proportion of sids randomly sampled for training. default=1.0')
parser.add_argument('--exp-name', default='debug_slice',
                    help='experiment name')

def main():
    # read configurations
    args = parser.parse_args()

    # define and create the experiment directory
    exp_dir = os.path.join('./ssl_exp', args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # save configurations to a dictionary
    with open(os.path.join(exp_dir, 'configs_slice.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    f.close()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        """
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        """

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print("Distributed:", args.distributed)

    #ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = args.npgus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.rank == 0:
        configure(os.path.join('./ssl_exp', args.exp_name))

    # create slice-level encoder
    if args.arch == 'custom':
        SliceNet = models.Encoder
    else:
        SliceNet = models.__dict__[args.arch]

    model_slice = MoCo_Slice(
        SliceNet,
        args.num_slice, args.rep_dim_slice, args.moco_dim_slice, args.moco_k_slice, args.moco_m_slice, args.moco_t_slice, args.mlp_slice)
    print(model_slice)
    print('Number of parameters: ' + str(count_parameters(model_slice)))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_slice.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size_slice = int(args.batch_size_slice / ngpus_per_node)
            args.workers_slice = int((args.workers_slice + ngpus_per_node - 1) / ngpus_per_node)
            model_slice = torch.nn.parallel.DistributedDataParallel(model_slice,
                                                                    device_ids=[args.gpu])
        else:
            raise NotImplementedError("GPU number is unknown.")
    else:
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer_slice = torch.optim.SGD(model_slice.parameters(), args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)
    # save the initial model
    if not args.resume_slice:
        if args.multiprocessing_distributed and args.rank % ngpus_per_node == 0:
            save_checkpoint({
                'epoch': 0,
                'arch': args.arch,
                'state_dict': model_slice.state_dict(),
                'optimizer': optimizer_slice.state_dict(),
            }, is_best=False,
                filename=os.path.join(os.path.join('./ssl_exp', args.exp_name), 'checkpoint_slice_init.pth.tar'))

    # optionally resume from a checkpoint
    if args.resume_slice:
        checkpoint = os.path.join('./ssl_exp', args.exp_name, args.resume_slice)
        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            if args.gpu is None:
                checkpoint_slice = torch.load(checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint_slice = torch.load(checkpoint, map_location=loc)
            args.start_epoch = checkpoint_slice['epoch']
            model_slice.load_state_dict(checkpoint_slice['state_dict'])
            optimizer_slice.load_state_dict(checkpoint_slice['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_slice, checkpoint_slice['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
            exit()

    # augmentation
    transform_resize = Resize(spatial_size=(args.slice_size, args.slice_size), mode='bilinear', align_corners=False)

    transform_flip_ax0 = RandFlip(spatial_axis=0, prob=0.2)
    transform_flip_ax1 = RandFlip(spatial_axis=1, prob=0.2)

    transform_re = Rand2DElastic(mode='bilinear', prob=1.0,
                                 spacing=(1.0, 1.0),
                                 #sigma_range=(8, 12),
                                 magnitude_range=(0, 1024 + 240),  # [-1024, 240] -> [0, 1024+240]
                                 #spatial_size=(args.slice_size, args.slice_size),
                                 translate_range=(14, 14),
                                 rotate_range=(np.pi / 12, np.pi / 12),
                                 scale_range=(0.1, 0.1),
                                 padding_mode='border'
                                 )

    transform_ra = RandAffine(mode='bilinear', prob=1.0,
                              #spatial_size=(args.slice_size, args.slice_size),
                              translate_range=(14, 14),
                              rotate_range=(np.pi / 12, np.pi / 12),
                              scale_range=(0.1, 0.1),
                              padding_mode='border')

    transform_rgn = RandGaussianNoise(prob=0.25, mean=0.0, std=50)
    transform_rac = RandAdjustContrast(prob=0.0)

    if args.transform_type == 'affine':
        train_transform = Compose([transform_resize, transform_flip_ax0, transform_flip_ax1, transform_rac, transform_rgn, transform_ra])
    if args.transform_type == 'elastic':
        train_transform = Compose([transform_resize, transform_flip_ax0, transform_flip_ax1, transform_rac, transform_rgn, transform_re])

    train_dataset_slice = COPD_dataset_slice("training", args, moco.loader.TwoCropsTransform(train_transform))
    args.num_sel_slices = len(train_dataset_slice.sel_slices)

    if args.distributed:
        train_sampler_slice = torch.utils.data.distributed.DistributedSampler(train_dataset_slice, shuffle=False) # unable random shuffle to ensure loop through all subjects
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    train_loader_slice = torch.utils.data.DataLoader(
        train_dataset_slice, batch_size=args.batch_size_slice, shuffle=(train_sampler_slice is None),
        num_workers=args.workers_slice, pin_memory=True, sampler=train_sampler_slice, drop_last=True)

    # define AverageMeter
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    AvgMeter_lst = [batch_time, data_time, losses, top1, top5]

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler_slice.set_epoch(epoch)
        adjust_learning_rate(optimizer_slice, epoch, args)
        # train for one epoch
        train_slice(train_loader_slice, model_slice, criterion, optimizer_slice, epoch, args, AvgMeter_lst)
        # save model for every epoch
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_slice.state_dict(),
                'optimizer': optimizer_slice.state_dict(),
            }, is_best=False, filename=os.path.join(os.path.join('./ssl_exp', args.exp_name),
                                                    'checkpoint_slice_{:04d}.pth.tar'.format(epoch + 1)))

def train_slice(train_loader, model, criterion, optimizer, epoch, args, AvgMeter_lst):

    [batch_time, data_time, losses, top1, top5] = AvgMeter_lst

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1))

    # switch to train mode
    model.train()
    end = time.time()
    num_iter_epoch = len(train_loader)
    #num_iter_sub_epoch = num_iter_epoch // args.num_slice
    num_iter_sub_epoch = num_iter_epoch // args.num_sel_slices
    print("num_iter_sub_epoch:", num_iter_sub_epoch)

    #slice_idx = -1
    # if epoch is even, loop slices from [low -> high]
    # if epoch is odd, loop slices reversely from [high -> low]
    # this helps to stabilize the negative pools
    if epoch % 2 == 0:
        sel_slices = train_loader.dataset.sel_slices
    if epoch % 2 == 1:
        sel_slices = train_loader.dataset.sel_slices
        sel_slices = sel_slices[::-1] # reverse

    j = -1
    for i, data in enumerate(train_loader, start=0):
        # measure data loading time
        data_time.update(time.time() - end)
        if i % num_iter_sub_epoch == 0:
            #slice_idx += 1
            #if slice_idx == args.num_slice:  # tail issue
            #    break
            j += 1
            if j == args.num_sel_slices:
                break
            slice_idx = sel_slices[j]
            train_loader.dataset.set_slice_idx(slice_idx)

        sids, images, slice_loc_idx, labels = data
        # one-hot encoding
        slice_idx_tensor = torch.zeros_like(slice_loc_idx)
        slice_idx_tensor = torch.add(slice_idx_tensor, slice_idx)
        #slice_one_hot = one_hot_embedding(slice_idx_tensor, args.num_slice)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            slice_loc_idx = slice_idx_tensor.long().cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(slice_idx=slice_idx, im_q=[images[0], slice_loc_idx], im_k=[images[1], slice_loc_idx])
        loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) and (i > 0):
            progress.display_slice(i, slice_idx)
            if args.rank == 0:
                step = i + num_iter_epoch * epoch
                log_value('slice/epoch', epoch, step)
                log_value('slice/loss', progress.meters[2].avg, step)
                log_value('slice/acc_1', progress.meters[3].avg, step)
                log_value('slice/acc_5', progress.meters[4].avg, step)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", slice_idx=0):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.slice_idx = slice_idx

    def display_slice(self, batch, slice_idx):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += ["slice :[{}]".format(slice_idx)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos_slice:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if (epoch + 1) >= (milestone * args.epochs) else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

if __name__ == '__main__':
    main()
