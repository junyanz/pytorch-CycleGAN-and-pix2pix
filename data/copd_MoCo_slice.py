from torch.utils.data import Dataset
import numpy as np
import glob

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, args, transforms=default_transform):
        self.args = args
        self.root_dir = args.root_dir
        self.metric_dict = dict() # initialize metric dictionary
        self.transforms = transforms
        self.slice_idx = 0
        self.slice_data = np.load('/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/slice_data_reg_mask/slice_'+str(self.slice_idx)+".npy") # 9201 * 447 * 447

        FILE = open("/ocean/projects/asc170022p/shared/Data/COPDGene/ClinicalData/phase 1 Final 10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
        mylist = FILE.readline().strip("\n").split("\t")
        metric_idx = [mylist.index(label) for label in self.args.label_name]
        race_idx = mylist.index("race")
        for line in FILE.readlines():
            mylist = line.strip("\n").split("\t")
            tmp = [mylist[idx] for idx in metric_idx]
            if "" in tmp:
                continue
            if self.args.nhw_only and mylist[race_idx] != "1":
                continue
            metric_list = []
            for i in range(len(metric_idx)):
                metric_list.append(float(tmp[i]))
            self.metric_dict[mylist[0]] = metric_list
        FILE.close()

        self.sid_list = []
        for item in glob.glob('/ocean/projects/asc170022p/lisun/registration/INSP2Atlas/image_transformed/'+"*.nii.gz"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-18])
        self.sid_list.sort()
        assert len(self.sid_list) == self.slice_data.shape[0]

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        self.sid_list_len = len(self.sid_list)
        print(stage+" dataset size:", self.sid_list_len)

    def set_slice_idx(self, idx):
        self.slice_idx = idx
        self.slice_data = np.load('/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/slice_data_reg_mask/slice_'+str(self.slice_idx)+".npy")

    def __len__(self):
        return self.sid_list_len*self.args.num_slice

    def __getitem__(self, idx):
        idx = idx % self.sid_list_len
        img = self.slice_data[idx,:,:]
        img = img + 1024.
        img = self.transforms(img[None,:,:])
        img[0] = img[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
        img[1] = img[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        key = self.sid_list[idx][:6]
        label = np.asarray(self.metric_dict[key]) # TODO: self.sid_list[idx][:6] extract sid from the first 6 letters

        return key, img, idx, label
