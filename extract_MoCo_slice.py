import os
import argparse
import json
from easydict import EasyDict as edict
from tqdm import tqdm

import random
import numpy as np
import torch
import math

from data.copd_MoCo_slice import COPD_dataset as COPD_dataset_slice

import models.cnn2d as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Extract 2D Slice Representations')
parser.add_argument('--exp-name', default='./ssl_exp/moco_slice_resnet18_224_512_128_mask_small')
parser.add_argument('--checkpoint-slice', default='checkpoint_slice_0001.pth.tar')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--slice-batch', type=int, default=128)


def main():
    # read configurations
    p = parser.parse_args()
    slice_epoch = p.checkpoint_slice.split('.')[0][-4:]
    with open(os.path.join(p.exp_name, 'configs_slice.json')) as f:
        args = edict(json.load(f))
    args.checkpoint = os.path.join(p.exp_name, p.checkpoint_slice)
    args.batch_size = p.batch_size
    args.slice_batch = p.slice_batch
    args.slice_rep_dir = os.path.join(p.exp_name, 'slice_rep', slice_epoch)
    os.makedirs(args.slice_rep_dir, exist_ok=True)

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    main_worker(args)

def main_worker(args):
    args.gpu = 0
    torch.cuda.set_device(args.gpu)
    loc = 'cuda:{}'.format(args.gpu)

    # create slice-level encoder
    if args.arch == 'custom':
        SliceNet = models.Encoder
    else:
        SliceNet = models.__dict__[args.arch]
    model_slice = SliceNet(rep_dim=args.rep_dim_slice, num_classes=args.moco_dim_slice)

    # remove the last FC layer
    model_slice.fc = torch.nn.Sequential()
    state_dict = torch.load(args.checkpoint, map_location=loc)['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    model_slice.load_state_dict(state_dict)
    print(model_slice)
    print("Slice model weights loaded.")
    model_slice.cuda()
    model_slice.eval()

    # dataset
    test_dataset_slice = COPD_dataset_slice('testing', args)
    test_loader = torch.utils.data.DataLoader(
        test_dataset_slice, batch_size=1, shuffle=False,
        num_workers=4, drop_last=False)
    args.label_name = args.label_name + args.label_name_set2
    args.num_slice = len(test_dataset_slice.sel_slices) # update number of slices to selected number of slices

    # test dataset
    sid_lst = []
    pred_arr = np.empty((len(test_dataset_slice), args.num_slice, args.rep_dim_slice))
    feature_arr = np.empty((len(test_dataset_slice), len(args.label_name) + len(args.visual_score) + len(args.P2_Pheno)))
    iterator = tqdm(test_loader,
                  desc="Propagating (X / X Steps)",
                  bar_format="{r_bar}",
                  dynamic_ncols=True,
                  disable=False)
    for i, batch in enumerate(iterator):
        sid, images, slice_loc_idx, adj, labels = batch
        sid_lst.append(sid[0])
        images = images[0].float().cuda()
        slice_loc_idx = slice_loc_idx.squeeze().long().cuda()
        with torch.no_grad():
            pred_lst = []
            num_iters = math.ceil(args.num_slice / args.slice_batch)
            for j in range(num_iters):
                if j < (num_iters - 1):
                    s = j * args.slice_batch
                    e = (j + 1) * args.slice_batch
                    pred = model_slice(images[s:e,:,:,:], slice_loc_idx[s:e])
                if j == (num_iters - 1):
                    s = j * args.slice_batch
                    pred = model_slice(images[s:, :, :, :], slice_loc_idx[s:])
                pred_lst.append(pred)
        pred_arr[i, :, :] = torch.cat(pred_lst).cpu().numpy()
        feature_arr[i:i + 1, :] = labels
        iterator.set_description("Propagating (%d / %d Steps)" % (i, len(test_dataset_slice)))
    np.save(os.path.join(args.slice_rep_dir, "sid_arr_full.npy"), sid_lst)
    np.save(os.path.join(args.slice_rep_dir, "pred_arr_slice_full.npy"), pred_arr)
    np.save(os.path.join(args.slice_rep_dir, "feature_arr_slice_full.npy"), feature_arr)
    print("\nExtraction slice representation on full set finished.")

if __name__ == '__main__':
    main()

