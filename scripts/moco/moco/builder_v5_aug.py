# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import random
import numpy as np

NUM_PATCHES = 581

def one_hot(idx, batch_size, gpu):
    a = torch.zeros(batch_size, NUM_PATCHES).cuda(gpu, non_blocking=True)
    a[:,idx] = 1
    return a

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, Unet3dPatchGenerator, gpu, num_patch, dim, K, m, T, mlp, model_A_ckpt, model_B_ckpt):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.num_locs = num_patch # TODO: add the new dimension of number of locations
        self.gpu = gpu

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self.model_A = Unet3dPatchGenerator(1, 1, n_blocks=9)
        self.model_B = Unet3dPatchGenerator(1, 1, n_blocks=9)
        loc = 'cpu'
        checkpoint_A = torch.load(model_A_ckpt, map_location=loc)
        self.model_A.load_state_dict(checkpoint_A)
        checkpoint_B = torch.load(model_B_ckpt, map_location=loc)
        self.model_B.load_state_dict(checkpoint_B)
        del checkpoint_A, checkpoint_B
        self.emph_median = np.load("/ocean/projects/asc170022p/rohit33/emphysemamedianvalues.npy")
        self.num_partition = 2

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K, self.num_locs, self.num_partition)) #TODO: the queue should be the size of (dim of reps) * (number of negative pairs) * (number of total locations)
        self.queue = nn.functional.normalize(self.queue, dim=0) # TODO: normalize patch representation

        self.register_buffer("queue_ptr", torch.zeros((self.num_locs, self.num_partition), dtype=torch.long)) # TODO: set pointer in buffer to 1 for each path location

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, patch_idx):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        emph_flag = keys[:,-1]
        keys = keys[:,:-1]

        ptr = self.queue_ptr
        #assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        for partition in range(self.num_partition):
            sub_keys = keys[emph_flag==partition]
            batch_size = sub_keys.shape[0]

            if ptr[patch_idx, partition].item() + batch_size <= self.K:
                self.queue[:, ptr[patch_idx, partition]:ptr[patch_idx, partition] + batch_size, patch_idx, partition] = sub_keys.T
                ptr[patch_idx, partition] = (ptr[patch_idx, partition] + batch_size) % self.K  # move pointer
            else:
                self.queue[:, ptr[patch_idx, partition]:, patch_idx, partition] = sub_keys.T[:,:self.K-ptr[patch_idx, partition]]
                ptr[patch_idx, partition] = 0

        self.queue_ptr = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, patch_idx, im_q, im_k, emph_flag):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        #print(0,patch_idx_one_hot.shape)
        # compute query features
        q = self.encoder_q(im_q[0], im_q[1])  # queries: NxC # TODO: encoder needs to take both pathces and their locations as inputs
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            im_k_neg = torch.empty(im_k[0].shape).cuda(self.gpu, non_blocking=True) # placeholder
            im_k_neg[emph_flag] = self.model_A(im_k[0][emph_flag], 2*im_k[1][emph_flag]-1) # low emphsema negative patch is send to model A
            emph_flag_neg = ~emph_flag
            im_k_neg[emph_flag_neg] = self.model_B(im_k[0][emph_flag_neg], 2*im_k[1][emph_flag_neg]-1) # high emphsema negative patch is send to model B

            k_neg = self.encoder_k(im_k_neg, im_k[1])  # keys: NxC
            k_neg = nn.functional.normalize(k_neg, dim=1)
            # dequeue and enqueue
            self._dequeue_and_enqueue(torch.cat([k_neg, emph_flag[:,None]], dim=1), patch_idx) # TODO: consider location for each patch in the batch

            im_k[0][emph_flag] = self.model_A(im_k[0][emph_flag], 2*im_k[1][emph_flag]-1)
            im_k[0][emph_flag] = self.model_B(im_k[0][emph_flag], 2*im_k[1][emph_flag]-1)
            im_k[0][emph_flag_neg] = self.model_B(im_k[0][emph_flag_neg], 2*im_k[1][emph_flag_neg]-1) # postive patch is send to model A and model B for reconstruction
            im_k[0][emph_flag_neg] = self.model_A(im_k[0][emph_flag_neg], 2*im_k[1][emph_flag_neg]-1) # postive patch is send to model A and model B for reconstruction

            # shuffle for making use of BN
            im_k[0], idx_unshuffle = self._batch_shuffle_ddp(im_k[0])

            k = self.encoder_k(im_k[0], im_k[1])  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        # TODO: compute negative logits for each path in the batch conditioned on their locations
        negs = torch.zeros((q.shape[0], self.queue.shape[0], self.queue.shape[1])).cuda(self.gpu, non_blocking=True)
        for i in range(self.num_partition):
            negs[emph_flag==i,:,:] = self.queue[:,:,patch_idx,1-i].clone().detach()[None,:,:]
        l_neg = torch.einsum('nc,nck->nk', [q, negs])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
