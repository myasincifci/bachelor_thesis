import os 
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ShortVideoDataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path)])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> torch.Tensor:
        image = Image.open(self.image_paths[index])

        if self.transform:
            image = self.transform(image)

        return image

if __name__ == '__main__':
    dataset = ShortVideoDataset('video_short')
    dataloader = DataLoader(dataset, 16)
    a=1