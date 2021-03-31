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

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from tensorboard_logger import configure, log_value


from models.cnn3d import Encoder as PatchNet
import moco.loader as MoCo_Loader
from moco.builder_Moco_patch import MoCo as MoCo_Patch
from data.copd_MoCo_patch import COPD_dataset as COPD_dataset_patch

from monai.transforms import Compose, ScaleIntensity, RandGaussianNoise, RandAffine, Rand3DElastic, RandAdjustContrast


parser = argparse.ArgumentParser(description='3D CT Images Self-Supervised Training Patch-level')
parser.add_argument('--arch', metavar='ARCH', default='custom')
parser.add_argument('--workers-patch', default=8, type=int, metavar='N',
                    help='patch-level number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size-patch', default=128, type=int,
                    metavar='N',
                    help='patch-level mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume-patch', default='', type=str, metavar='PATH',
                    help='path to latest patch-level checkpoint (default: None)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
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

# image data configs:
parser.add_argument('--stage', default='training', type=str,
                    help='stage: training or testing')
parser.add_argument('--num-patch', default=581, type=int,
                    help='total number of patches in the atlas image.')
parser.add_argument('--root-dir', default='/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/patch_data_32_6_reg_mask/',
                    help='root directory of registered images in COPD dataset')
parser.add_argument('--label-name', default=["FEV1pp_utah", "FEV1_FVC_utah", "finalGold"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--label-name-set2', default=["Exacerbation_Frequency", "MMRCDyspneaScor"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--visual-score', default=["Emph_Severity", "Emph_Paraseptal"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--P2-Pheno', default=["Exacerbation_Frequency_P2"], nargs='+',
                    help='phenotype label names')
parser.add_argument('--nhw-only', action='store_true',
                    help='only include white people')
parser.add_argument('--fold', default=0, type=int,
                    help='fold index of cross validation')

# MoCo specific configs:
parser.add_argument('--rep-dim-patch', default=512, type=int,
                    help='feature dimension (default: 512)')
parser.add_argument('--moco-dim-patch', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k-patch', default=4096, type=int,
                    help='queue size; number of negative keys (default: 4098)')
parser.add_argument('--moco-m-patch', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t-patch', default=0.2, type=float,
                    help='softmax temperature (default: 0.2)')

# options for moco v2
parser.add_argument('--mlp-patch', action='store_false',
                    help='use mlp head')
parser.add_argument('--cos-patch', action='store_false',
                    help='use cosine lr schedule')

# experiment configs
parser.add_argument('--transform-type', default='affine', type=str,
                    help='image transformation type, affine or elastic (default: affine)')
parser.add_argument('--exp-name', default='debug_patch', type=str,
                    help='experiment name')


def main():
    # read configurations
    args = parser.parse_args()

    # define and create the experiment directory
    exp_dir = os.path.join('./ssl_exp', args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)

    # save configurations to a dictionary
    with open(os.path.join(exp_dir, 'configs_patch.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    f.close()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        #cudnn.deterministic = True

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

    # create patch-level encoder
    model_patch = MoCo_Patch(
        PatchNet,
        args.num_patch, args.rep_dim_patch, args.moco_dim_patch, args.moco_k_patch, args.moco_m_patch, args.moco_t_patch, args.mlp_patch)
    print(model_patch)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_patch.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size_patch = int(args.batch_size_patch / ngpus_per_node)
            args.workers_patch = int((args.workers_patch + ngpus_per_node - 1) / ngpus_per_node)
            model_patch = torch.nn.parallel.DistributedDataParallel(model_patch,
                                                                    device_ids=[args.gpu])
        else:
            raise NotImplementedError("GPU number is unknown.")
    else:
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer_patch = torch.optim.SGD(model_patch.parameters(), args.lr,
                                      momentum=args.momentum,
                                      weight_decay=args.weight_decay)
    # save the initial model
    if not args.resume_patch:
        if args.multiprocessing_distributed and args.rank % ngpus_per_node == 0:
            save_checkpoint({
                'epoch': 0,
                'arch': args.arch,
                'state_dict': model_patch.state_dict(),
                'optimizer': optimizer_patch.state_dict(),
            }, is_best=False,
                filename=os.path.join(os.path.join('./ssl_exp', args.exp_name), 'checkpoint_patch_init.pth.tar'))

    # optionally resume from a checkpoint
    if args.resume_patch:
        checkpoint = os.path.join('./ssl_exp', args.exp_name, args.resume_patch)
        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            if args.gpu is None:
                checkpoint_patch = torch.load(checkpoint)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint_patch = torch.load(checkpoint, map_location=loc)
            args.start_epoch = checkpoint_patch['epoch']
            model_patch.load_state_dict(checkpoint_patch['state_dict'])
            optimizer_patch.load_state_dict(checkpoint_patch['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume_patch, checkpoint_patch['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))
            exit()

    # augmentation
    transform_ra = RandAffine(mode='bilinear', prob=1.0,
                              spatial_size=(32, 32, 32),
                              translate_range=(12, 12, 12),
                              rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                              scale_range=(0.1, 0.1, 0.1),
                              padding_mode='border')

    transform_re = Rand3DElastic(mode='bilinear', prob=1.0,
                                 sigma_range=(8, 12),
                                 magnitude_range=(0, 1024 + 240),  # [-1024, 240] -> [0, 1024+240]
                                 spatial_size=(32, 32, 32),
                                 translate_range=(12, 12, 12),
                                 rotate_range=(np.pi / 18, np.pi / 18, np.pi / 18),
                                 scale_range=(0.1, 0.1, 0.1),
                                 padding_mode='border')

    transform_rgn = RandGaussianNoise(prob=0.25, mean=0.0, std=50)
    transform_rac = RandAdjustContrast(prob=0.25)

    if args.transform_type == 'affine':
        train_transforms = Compose([transform_rac, transform_rgn, transform_ra])
    if args.transform_type == 'elastic':
        train_transforms = Compose([transform_rac, transform_rgn, transform_re])

    train_dataset_patch = COPD_dataset_patch('training', args, MoCo_Loader.TwoCropsTransform(train_transforms))


    if args.distributed:
        train_sampler_patch = torch.utils.data.distributed.DistributedSampler(train_dataset_patch)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    train_loader_patch = torch.utils.data.DataLoader(
        train_dataset_patch, batch_size=args.batch_size_patch, shuffle=(train_sampler_patch is None),
        num_workers=args.workers_patch, pin_memory=True, sampler=train_sampler_patch, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler_patch.set_epoch(epoch)
        adjust_learning_rate(optimizer_patch, epoch, args)
        # train for one epoch
        train_patch(train_loader_patch, model_patch, criterion, optimizer_patch, epoch, args)
        # save model for every epoch
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model_patch.state_dict(),
                'optimizer': optimizer_patch.state_dict(),
            }, is_best=False, filename=os.path.join(os.path.join('./ssl_exp', args.exp_name),
                                                    'checkpoint_patch_{:04d}.pth.tar'.format(epoch + 1)))

def train_patch(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch + 1))

    # switch to train mode
    model.train()
    end = time.time()
    num_iter_epoch = len(train_loader)
    num_iter_sub_epoch = num_iter_epoch // args.num_patch
    print("num_iter_sub_epoch:", num_iter_sub_epoch)

    patch_idx = -1
    for i, data in enumerate(train_loader, start=0):
        # measure data loading time
        data_time.update(time.time() - end)
        if i % num_iter_sub_epoch == 0:
            patch_idx += 1
            if patch_idx == args.num_patch:  # tail issue
                break
            train_loader.dataset.set_patch_idx(patch_idx)

        pid, images, patch_loc_idx, adj, label = data

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            patch_loc_idx = patch_loc_idx.float().cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(patch_idx=patch_idx, im_q=[images[0], patch_loc_idx], im_k=[images[1], patch_loc_idx])
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
            progress.display_patch(i, patch_idx)
            if args.rank == 0:
                step = i + num_iter_epoch * epoch
                log_value('patch/epoch', epoch, step)
                log_value('patch/loss', progress.meters[2].avg, step)
                log_value('patch/acc_1', progress.meters[3].avg, step)
                log_value('patch/acc_5', progress.meters[4].avg, step)


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
    def __init__(self, num_batches, meters, prefix="", patch_idx=0):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.patch_idx = patch_idx

    def display_patch(self, batch, patch_idx):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += ["Patch :[{}]".format(patch_idx)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos_patch:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
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

if __name__ == '__main__':
    main()
