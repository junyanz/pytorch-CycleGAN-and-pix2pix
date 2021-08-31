from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import random
from monai.transforms import Resize, RandAffine

DATA_DIR = "/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalData/"

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, args, transforms=default_transform):
        self.stage = stage
        self.args = args
        self.root_dir = args.root_dir
        self.metric_dict = dict() # initialize metric dictionary
        self.transforms = transforms
        self.slice_idx = 0
        self.slice_data = np.load('/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/slice_data_reg_mask/slice_'+str(self.slice_idx)+".npy") # 9201 * 447 * 447
        self.mask_data = np.load('/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/slice_mask_reg_mask/slice_' + str(self.slice_idx)+'.npz')['arr_0']

        # lung mask selection criteria
        self.slice_mask_summary = pd.read_csv('/ocean/projects/asc170022p/yuke/PythonProject/pytorch-CycleGAN-and-pix2pix/copd_slice_lung_mask_summary.csv')
        sel_idx = self.slice_mask_summary['p50_prop'] > args.mask_threshold
        self.sel_slices = self.slice_mask_summary[sel_idx]['slice'].tolist()

        if stage == 'training':
            FILE = open(DATA_DIR + "phase 1 Final 10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.args.label_name]
            sid_set = set()
            race_idx = mylist.index("race")
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                tmp = [mylist[idx] for idx in metric_idx]
                if "" in tmp:
                    continue
                sid_set.add(mylist[0])
                if self.args.nhw_only and mylist[race_idx] != "1":
                    continue
                metric_list = []
                for i in range(len(metric_idx)):
                    metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]] = metric_list
            FILE.close()

        if stage == 'testing':
            self.label_name = self.args.label_name + self.args.label_name_set2
            FILE = open(DATA_DIR + "phase 1 Final 10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.label_name]
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                tmp = [mylist[idx] for idx in metric_idx]
                if "" in tmp[:3]:
                    continue
                metric_list = []
                for i in range(len(metric_idx)):
                    if tmp[i] == "":
                        metric_list.append(-1024)
                    else:
                        metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]] = metric_list + [-1024, -1024, -1024]
            FILE.close()

            FILE = open(DATA_DIR + "CT scan datasets/CT visual scoring/COPDGene_CT_Visual_20JUL17.txt", "r")
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.args.visual_score]
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                if mylist[0] not in self.metric_dict:
                    continue
                tmp = [mylist[idx] for idx in metric_idx]
                metric_list = []
                for i in range(len(metric_idx)):
                    metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]][
                -len(self.args.visual_score) - len(self.args.P2_Pheno):-len(self.args.P2_Pheno)] = metric_list
            FILE.close()

            FILE = open(
                DATA_DIR + 'P1-P2 First 5K Long Data/Subject-flattened- one row per subject/First5000_P1P2_Pheno_Flat24sep16.txt',
                'r')
            mylist = FILE.readline().strip("\n").split("\t")
            metric_idx = [mylist.index(label) for label in self.args.P2_Pheno]
            for line in FILE.readlines():
                mylist = line.strip("\n").split("\t")
                if mylist[0] not in self.metric_dict:
                    continue
                tmp = [mylist[idx] for idx in metric_idx]
                metric_list = []
                for i in range(len(metric_idx)):
                    metric_list.append(float(tmp[i]))
                self.metric_dict[mylist[0]][-len(self.args.P2_Pheno):] = metric_list
            FILE.close()

        self.sid_list = []
        for item in glob.glob('/ocean/projects/asc170022p/lisun/registration/INSP2Atlas/image_transformed/'+"*.nii.gz"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-7])
        self.sid_list.sort()
        assert len(self.sid_list) == self.slice_data.shape[0]

        # handel outliers
        self.outliers = [
                '25369H_INSP_STD_NJC_COPD_Reg_19676E',
                '22169K_INSP_STD_TXS_COPD_Reg_19676E',
                '15320X_INSP_STD_TEM_COPD_Reg_19676E',
                '22225U_INSP_STD_NJC_COPD_Reg_19676E',
                '17369L_INSP_STD_COL_COPD_Reg_19676E',
                '19642N_INSP_STD_NJC_COPD_Reg_19676E',
                '22652N_INSP_STD_JHU_COPD_Reg_19676E',
                '18597D_INSP_STD_TEM_COPD_Reg_19676E',
                '18489A_INSP_STD_UIA_COPD_Reg_19676E',
                '18677B_INSP_STD_HAR_COPD_Reg_19676E',
                '14457T_INSP_STD_NJC_COPD_Reg_19676E',
                '23116U_INSP_STD_UIA_COPD_Reg_19676E',
                '23163D_INSP_STD_FAL_COPD_Reg_19676E',
                '23372M_INSP_STD_UIA_COPD_Reg_19676E',
                '24634V_INSP_STD_UIA_COPD_Reg_19676E',
                '23761X_INSP_STD_UIA_COPD_Reg_19676E',
                '11500F_INSP_STD_HAR_COPD_Reg_19676E'
               ]
        idx_o = []
        for sid in self.sid_list:
            if sid in self.outliers:
                idx_o.append(True)
            else:
                idx_o.append(False)
        self.idx_o = np.array(idx_o)
        # remove outliers
        self.sid_list = np.asarray(self.sid_list)
        self.sid_list = self.sid_list[~self.idx_o]

        # handel sub-sample
        if args.sample_prop < 1.0:
            # random seed
            random.seed(args.seed)
            self.sid_idx = random.sample(range(0, len(self.sid_list)), int(len(self.sid_list) * args.sample_prop))
            self.sid_idx.sort()
            self.sid_list = list(self.sid_list[i] for i in self.sid_idx) # select sampled sids

        if stage == 'training':
            np.save(os.path.join('./ssl_exp', args.exp_name, 'sid_list.npy'), self.sid_list)
        if stage == 'testing':
            sid_list = np.load(os.path.join('./ssl_exp', args.exp_name, 'sid_list.npy'))
            assert (sid_list == self.sid_list).sum() == len(sid_list) # check if the sid lists are consistent

        print("Fold: full")
        self.sid_list_len = len(self.sid_list)
        print(stage+" dataset size:", self.sid_list_len)

    def set_slice_idx(self, idx):
        self.slice_idx = idx
        self.slice_data = np.load('/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/slice_data_reg_mask/slice_'+str(self.slice_idx)+'.npy')
        self.mask_data = np.load('/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/slice_mask_reg_mask/slice_' + str(self.slice_idx)+'.npz')['arr_0']

        # remove outliers
        self.slice_data = self.slice_data[~self.idx_o]
        self.mask_data = self.mask_data[~self.idx_o]

        if self.args.sample_prop < 1.0: # select sampled sids
            self.slice_data = self.slice_data[self.sid_idx]
            self.mask_data = self.mask_data[self.sid_idx]

    def __len__(self):
        if self.stage == 'training':
            # make sure the total samples is an int * batch size
            n = len(self.sid_list) // (self.args.batch_size_slice * self.args.npgus_per_node)
            return n * self.args.npgus_per_node * self.args.batch_size_slice * len(self.sel_slices)
        if self.stage == 'testing':
            return self.sid_list_len

    def __getitem__(self, idx):
        if self.stage == 'training':
            idx = idx % self.sid_list_len #TODO: this could result duplicates, e.g., idx = 1, idx = self.sid_list_len + 1
            img = self.slice_data[idx,:,:] # self.slice_data.shape = 9201 * 447 * 447
            mask = self.mask_data[idx,:,:] # binary lung mask 9201 * 447 * 447

            # set region outside lung mask (False) = -1024
            if self.args.mask_imputation:
                img[~mask] = -1024

            # resize the input image
            transform_resize = Resize(spatial_size=(self.args.slice_size, self.args.slice_size), mode='bilinear',
                                      align_corners=False)
            img = transform_resize(img[None, :, :])

            # preprocess input image
            img = np.clip(img, -1024, 240)  # clip input intensity to [-1024, 240]
            img = img + 1024.
            img_lst = self.transforms(img) # return the augmented anchor img and its positive pair

            # normalize pixel values
            img_lst[0] = img_lst[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
            img_lst[1] = img_lst[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

            key = self.sid_list[idx][:6]
            label = np.asarray(self.metric_dict[key]) # TODO: self.sid_list[idx][:6] extract sid from the first 6 letters
            return key, img_lst, mask, idx, label

        if self.stage == 'testing':
            sid = self.sid_list[idx]

            # load subject-level images (nifti format)
            img = sitk.ReadImage('/ocean/projects/asc170022p/lisun/registration/INSP2Atlas/image_transformed/' + sid + ".nii.gz")
            img = sitk.GetArrayFromImage(img)

            mask = sitk.ReadImage('/ocean/projects/asc170022p/lisun/registration/INSP2Atlas/unet_mask_transformed/' + sid + '_Affine.nii.gz')
            mask = sitk.GetArrayFromImage(mask)

            # select slices with lung mask region
            img = img[self.sel_slices]
            mask = mask[self.sel_slices]

            # set region outside lung mask (False) = -1024
            if self.args.mask_imputation:
                img = np.where(mask != 0, img, -1024)

            # resize the input image
            transform_resize = Resize(spatial_size=(self.args.slice_size, self.args.slice_size), mode='bilinear',
                                      align_corners=False)
            img = transform_resize(img[None, :, :])

            # preprocess input image
            img = np.clip(img, -1024, 240)  # clip input intensity to [-1024, 240]
            img = img + 1024.
            img = self.transforms(img)

            # normalize pixel values
            img = img[:, None, :, :] / 632. - 1  # Normalize to [-1,1], 632=(1024+240)/2

            key = self.sid_list[idx][:6]
            label = np.asarray(self.metric_dict[key])  # extract sid from the first 6 letters

            #adj = np.load(self.root_dir + "adj/" + sid + "_adj.npy")
            #adj = (adj > 0.13).astype(np.int)
            adj = np.array([])  # not needed for patch-level only

            # return selected slice index
            slice_seq = np.array(self.sel_slices)
            return sid, img, mask, slice_seq, adj, label

