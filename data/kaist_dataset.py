from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

# 내가 새로 만든 import file들
from pathlib import Path

class KaistDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.dir_path = Path(opt.dataroot) / "images"
        self.lwir_dataset, self.visible_dataset = self.__get_image_paths(self.dir_path, opt.max_dataset_size)
        self.transform = get_transform(opt)


    def __get_image_paths(self, root_path, max_size):
        lwir_dataset, visible_dataset = [], []

        for set_dir in sorted(root_path.iterdir()):
            if not set_dir.is_dir(): continue

            for v_dir in sorted(set_dir.iterdir()):
                if not v_dir.is_dir(): continue

                lwir_path = v_dir / 'lwir'
                visible_path = v_dir / "visible"

                if not lwir_path.exists() :
                    raise FileNotFoundError(f"lwir does not exist on {v_dir}")
                if not visible_path.exists() :
                    raise FileNotFoundError(f"visible does not exist on {v_dir}")

                cur_lwir_dataset = sorted(make_dataset(lwir_path, max_size))
                cur_visible_dataset = sorted(make_dataset(lwir_path, max_size)) 

                for lwir_image, visible_image in zip(cur_lwir_dataset, cur_visible_dataset):
                    lwir_dataset.append(lwir_image)
                    visible_dataset.append(visible_image)
                    max_size -= 1
                    if max_size <= 0:
                        return lwir_dataset, visible_dataset
        return lwir_dataset, visible_dataset
    

    def __getitem__(self, index):
        lwir_path = self.lwir_dataset[index]
        visible_path = self.visible_dataset[index]
        
        lwir_image = Image.open(lwir_path).convert('RGB')
        visible_image = Image.open(visible_path).convert('RGB')
        
        lwir_data = self.transform(lwir_image)
        visible_data = self.transform(visible_image)
        
        return {'A': lwir_data, 'B': visible_data, 'A_paths': lwir_path, 'B_paths': visible_path}

    def __len__(self):
        return len(self.lwir_dataset)