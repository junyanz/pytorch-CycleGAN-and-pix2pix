from torch.utils.data import Dataset
import numpy as np
import glob

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
        self.patch_idx = 0
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(self.patch_idx)+".npy")

        if stage == 'training':
            FILE = open(DATA_DIR + "phase 1 Final 10K/phase 1 Pheno/Final10000_Phase1_Rev_28oct16.txt", "r")
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
        for item in glob.glob(self.args.root_dir+"patch/"+"*_patch.npy"):
            if item.split('/')[-1][:6] not in self.metric_dict:
                continue
            self.sid_list.append(item.split('/')[-1][:-10])
        self.sid_list.sort()
        assert len(self.sid_list) == self.patch_data.shape[0]
        self.patch_loc = np.load(self.args.root_dir + "19676E_INSP_STD_JHU_COPD_BSpline_Iso1_patch_loc.npy")
        self.patch_loc = (self.patch_loc / self.patch_loc.max(0)) * 2 - 1  # TODO: normalize position to [-1, 1]

        print("Fold: full")
        self.sid_list = np.asarray(self.sid_list)
        self.sid_list_len = len(self.sid_list)
        print(stage+" dataset size:", self.sid_list_len)

    def set_patch_idx(self, idx):
        self.patch_idx = idx
        self.patch_data = np.load(self.args.root_dir+"grouped_patch/patch_loc_"+str(idx)+".npy")

    def __len__(self):
        if self.stage == 'training':
            return self.sid_list_len*self.args.num_patch
        if self.stage == 'testing':
            return self.sid_list_len

    def __getitem__(self, idx):

        if self.stage == 'training':
            idx = idx % self.sid_list_len
            img = self.patch_data[idx,:,:,:]
            img = np.clip(img, -1024, 240) # clip input intensity to [-1024, 240]
            img = img + 1024.
            img = self.transforms(img[None,:,:,:])
            img[0] = img[0]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2
            img[1] = img[1]/632.-1 # Normalize to [-1,1], 632=(1024+240)/2

            patch_loc_idx = self.patch_loc[self.patch_idx,:] # patch location
            adj = np.array([]) # not needed for patch-level only

            key = self.sid_list[idx][:6]
            label = np.asarray(self.metric_dict[key])
            return key, img, patch_loc_idx, adj, label

        if self.stage == 'testing':
            sid = self.sid_list[idx]
            img = np.load(self.root_dir + "patch/" + sid + "_patch.npy")
            img = np.clip(img, -1024, 240)  # clip input intensity to [-1024, 240]
            img = img + 1024.
            img = self.transforms(img)
            img = img[:, None, :, :, :] / 632. - 1  # Normalize to [-1,1], 632=(1024+240)/2

            key = self.sid_list[idx][:6]
            label = np.asarray(self.metric_dict[key])  # extract sid from the first 6 letters

            #adj = np.load(self.root_dir + "adj/" + sid + "_adj.npy")
            #adj = (adj > 0.13).astype(np.int)
            adj = np.array([])  # not needed for patch-level only
            return sid, img, self.patch_loc.copy(), adj, label