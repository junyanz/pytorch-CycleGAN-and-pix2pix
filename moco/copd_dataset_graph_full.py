from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import numpy as np
import glob

def default_transform(x):
    return x

class COPD_dataset(Dataset):

    def __init__(self, stage, cfg, transforms=default_transform):
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.metric_dict = dict() # initialize metric dictionary
        self.transforms = transforms

        FILE = open("./misc/Final10000_Phase1_Rev_28oct16.txt", "r")
        mylist = FILE.readline().strip("\n").split("\t")
        metric_idx = [mylist.index(label) for label in self.cfg.label_name]
        race_idx = mylist.index("race")
        for line in FILE.readlines():
            mylist = line.strip("\n").split("\t")
            tmp = [mylist[idx] for idx in metric_idx]
            if "" in tmp:
                continue
            metric_list = []
            for i in range(len(metric_idx)):
                metric_list.append(float(tmp[i]))
            self.metric_dict[mylist[0]] = metric_list
        FILE.close()

        self.sid_list = []
        for item in glob.glob(self.cfg.root_dir+"patch/"+"*_patch.npy"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        self.patch_loc = np.load("../data/patch_data_32_6_reg/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0) # column-wise norm
        self.adj = self.prep_adj()

        print("Fold: Full")
        self.sid_list = np.asarray(self.sid_list)
        print(stage+" dataset size:", len(self))

    def __len__(self):
        return len(self.sid_list)

    def prep_adj(self):
        adj = np.load(self.root_dir+"adj/19676E_INSP_STD_JHU_COPD_adj.npy") # unique: [0., 0.0065918 , 0.03515625, 0.1875, 1.]
        adj=(adj>0.18).astype(np.int)
        return adj

    def __getitem__(self, idx):
        img = np.load(self.root_dir+"patch/"+self.sid_list[idx]+"_patch.npy")
        img = img + 1024.
        img = self.transforms(img)
        img[0] = img[0][:,None,:,:,:]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
        img[1] = img[1][:,None,:,:,:]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        return img, self.patch_loc.copy(), self.adj.copy()
