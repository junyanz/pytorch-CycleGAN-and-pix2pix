import torch.utils.data as data

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        
    def name(self):
        return 'BaseDataset'
    
    def initialize(self, opt):
        pass

