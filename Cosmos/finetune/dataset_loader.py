import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, path, mode):
        super().__init__()
        self.mode = mode
        self.size = 256
        self.fix_depth = 120

        with open(path, 'r') as f:
            self.data = [line.strip() for line in f.readlines()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        npy_path = self.data[idx]
        sample = np.load(npy_path)  # shape: (D_orig, H, W)
        if len(sample.shape) == 4:
            sample = sample.squeeze(0)
        sample = torch.from_numpy(sample).float() / 32767.0  # (D_orig, H, W)
        
        sample = sample.unsqueeze(0).unsqueeze(0)
        sample = F.interpolate(sample, size=(self.fix_depth, self.size, self.size), mode='trilinear', align_corners=False)

        # Remove batch dim, keep channel dim → Shape becomes (1, D, H, W)
        sample = sample.squeeze(0)  # Keep (C, D, H, W)

        # Expand to (D, H, W, 3) if needed
        sample = sample.permute(1, 2, 3, 0).expand(-1, -1, -1, 3)  # (D, H, W, 3)

        return sample  # Correct shape: (D, H, W, 3)

def get_dataloader(path_to_txt, mode="train", batch_size=4, shuffle=True, num_workers=4):
    dataset = CustomDataset(path=path_to_txt, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if mode == "train" else False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
