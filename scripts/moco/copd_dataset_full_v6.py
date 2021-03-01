from sklearn.model_selection import KFold
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
        self.list_exist_mask = np.load(self.args.root_dir+"list_exist_mask.npy") == 1
        self.patch_idx = 0
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.patch_idx)+".npy")[self.list_exist_mask,:]
        self.pct_emph_data = np.load(self.args.root_dir+"grouped_pct_emph/patch_loc_"+str(self.patch_idx)+".npy")
        self.emph_median = np.load("/ocean/projects/asc170022p/rohit33/emphysemamedianvalues.npy")

        FILE = open("./misc/Final10000_Phase1_Rev_28oct16.txt", "r")
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
        #assert len(self.sid_list) == self.patch_data.shape[0]
        assert self.patch_data.shape[0] == self.pct_emph_data.shape[0]
        self.patch_loc = np.load("../data/patch_data_32_6_reg_mask/19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = self.patch_loc / self.patch_loc.max(0)

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        #self.sid_list_len = len(self.sid_list)
        self.sid_list_len = self.patch_data.shape[0]
        print(stage+" dataset size:", self.sid_list_len)

    def set_patch_idx(self, idx):
        self.patch_idx = idx
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(idx)+".npy")[self.list_exist_mask,:]
        self.pct_emph_data = np.load(self.args.root_dir+"grouped_pct_emph/patch_loc_"+str(idx)+".npy")

    def __len__(self):
        return self.sid_list_len*self.args.num_patch

    def __getitem__(self, idx):
        idx = idx % self.sid_list_len
        img = self.patch_data[idx,:,:,:]
        pct_emph = self.pct_emph_data[idx,0]
        emph_flag = pct_emph < self.emph_median[self.patch_idx]

        img = img + 1024.
        img = self.transforms(img[None,:,:,:])
        img[0] = img[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
        img[1] = img[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

        patch_loc_idx = self.patch_loc[self.patch_idx,:] # patch location
        adj = np.array([]) # not needed for patch-level only

        #adj = np.load(self.root_dir+"adj/"+self.sid_list[idx]+"_adj.npy") # TODO: adj matrix
        #adj = (adj == 1).astype(np.int)
        #adj = np.logical_and(adj >= 1, adj < 1.42).astype(np.int)
        #adj = adj + np.eye(adj.shape[0])
        #keep_idx = np.sum(adj, 0) > 0
        #adj = adj[keep_idx,:] # Remove isolate patches
        #adj = adj[:,keep_idx]
        #adj = (adj / np.sum(adj, 0)).transpose()

        key = self.sid_list[idx][:6]
        label = np.asarray(self.metric_dict[key]) # TODO: self.sid_list[idx][:6] extract sid from the first 6 letters

        return key, img, patch_loc_idx, emph_flag, adj, label
