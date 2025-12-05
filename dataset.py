import torch 
from torch.utils.data import Dataset
import cv2
import numpy as np

class bone(Dataset):
    def __init__(self, df, size=224):
        self.df = df
        self.size = size

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample = self.df.iloc[index]

        img_path  = sample["xrays"]
        mask_path = sample.get("masks", None)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0

        if mask_path is None or isinstance(mask_path, float) or mask_path == "":
            mask = torch.zeros((1, self.size, self.size), dtype=torch.float32)
            return img, mask

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask
