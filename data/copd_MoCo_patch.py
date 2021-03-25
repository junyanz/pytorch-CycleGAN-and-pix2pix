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
        self.patch_idx = 0
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.patch_idx)+".npy")

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
        for item in glob.glob(self.args.root_dir+"patch/"+"*_patch.npy"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        assert len(self.sid_list) == self.patch_data.shape[0]
        self.patch_loc = np.load(
            "/ocean/projects/asc170022p/lisun/copd/gnn_shared/data/patch_data_32_6_reg_mask/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0)

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        self.sid_list_len = len(self.sid_list)
        print(stage+" dataset size:", self.sid_list_len)

    def set_patch_idx(self, idx):
        self.patch_idx = idx
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(idx)+".npy")

    def init_bow_dict(self):
        bow_dict = dict()
        FILE = open('/ocean/projects/asc170022p/shared/Data/COPDGene/CuratedFeatures/Nonparametric_SuperVoxel/BOW_histFHOG_largeRange_setting1.csv', 'r')
        columns = FILE.readline().strip("\n").split(",")
        for line in FILE.readlines():
            mylist = line.strip("\n").split(",")
            key = mylist[-1] # key is the sid
            bow = list(map(float, mylist[:-1]))
            bow_dict[key] = np.array(bow)
        return bow_dict # TODO: total sids = 7292

    def init_lle_dict(self):
        lle_dict = dict()
        FILE = open(
            '/ocean/projects/asc170022p/shared/Data/COPDGene/CuratedFeatures/Nonparametric_SuperVoxel/LLE_histFHOG_largeRange_setting1.csv', 'r')
        columns = FILE.readline().strip("\n").split(",")
        for line in FILE.readlines():
            mylist = line.strip("\n").split(",")
            key = mylist[-1]  # key is the sid
            lle = list(map(float, mylist[:-1]))
            lle_dict[key] = np.array(lle)
        return lle_dict # TODO: total sids = 7292


    def __len__(self):
        return self.sid_list_len*self.args.num_patch

    def __getitem__(self, idx):
        idx = idx % self.sid_list_len
        img = self.patch_data[idx,:,:,:]
        img = img + 1024.
        img = self.transforms(img[None,:,:,:])
        img[0] = img[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
        img[1] = img[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        patch_loc_idx = self.patch_loc[self.patch_idx,:] # patch location
        adj = np.array([]) # not needed for patch-level only

        key = self.sid_list[idx][:6]
        label = np.asarray(self.metric_dict[key]) # TODO: self.sid_list[idx][:6] extract sid from the first 6 letters

        return key, img, patch_loc_idx, adj, label
