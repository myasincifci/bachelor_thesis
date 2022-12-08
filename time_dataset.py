import os 
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt

class ShortVideoDatasetTime(Dataset):
    def __init__(self, path, transform=None) -> None:
        self.transform = transform
        self.image_paths = sorted([os.path.join(path, p) for p in os.listdir(path)])
        
        self.mean = None
        for path in self.image_paths:

            image = Image.open(path)
            image = self.transform(image)

            if self.mean == None:
                self.mean = image
            else:
                self.mean += image
        self.mean /= len(self.image_paths)

        self.std = None
        for path in self.image_paths:

            image = Image.open(path)
            image = self.transform(image)

            if self.std == None:
                self.std = (image - self.mean)**2
            else:
                self.std += (image - self.mean)**2
        self.std /= len(self.image_paths)
        self.std = torch.sqrt(self.std)


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index, training=True) -> torch.Tensor:
        image = Image.open(self.image_paths[index])

        if self.transform:
            image = self.transform(image)

        if training:
            image = (image - self.mean)/self.std

        return (index, image)

if __name__ == '__main__':
    dataset = ShortVideoDatasetTime('video_short')
    dataloader = DataLoader(dataset, 16)
    a=1