from short_video_dataset import ShortVideoDataset

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from lightly.transforms import GaussianBlur
from lightly.transforms import RandomRotate

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np

from typing import Tuple

input_size: int = 256
cj_prob: float = 0.8
cj_bright: float = 0.7
cj_contrast: float = 0.7
cj_sat: float = 0.7
cj_hue: float = 0.2
min_scale: float = 0.15
random_gray_scale: float = 0.2
gaussian_blur: float = 0.5
kernel_size: float = 0.01
vf_prob: float = 0.0
hf_prob: float = 0.5
rr_prob: float = 0.0

color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )

transform = T.Compose([
                T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
                T.CenterCrop(size=input_size),
                RandomRotate(prob=rr_prob),
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomVerticalFlip(p=vf_prob),
                T.RandomApply([color_jitter], p=cj_prob),
                T.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(kernel_size=int(kernel_size * input_size), prob=gaussian_blur),
                T.ToTensor()
            ])

def apply_transforms(batch: torch.Tensor) -> torch.Tensor():

    # resize = T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
    resize = T.Resize(input_size)
    ccrop = T.CenterCrop(size=batch[0].shape[1] * (np.random.rand()*0.3 + 0.7))
    rotate = RandomRotate(prob=rr_prob)
    hflip = T.RandomHorizontalFlip()
    vflip = T.RandomVerticalFlip(p=vf_prob)
    jitter = T.RandomApply([color_jitter], p=cj_prob)
    #grayscale = T.RandomGrayscale(p=random_gray_scale)
    # blur = GaussianBlur(kernel_size=kernel_size * input_size, prob=gaussian_blur)
    blur = T.GaussianBlur(kernel_size=3)#int(kernel_size * input_size))

    batch_ = T.Grayscale()(batch)

    batch_ = ccrop(batch_)
    batch_ = resize(batch_)

    # if np.random.rand() < hf_prob:
    #     batch_ = hflip(batch_)  

    if np.random.rand() < cj_prob:
        batch_ = jitter(batch_)    
    
    # if np.random.rand() < random_gray_scale:
    #     batch_ = grayscale(batch_)
    
    if np.random.rand() < gaussian_blur:
        batch_ = blur(batch_)

    return batch_

def collate_fnc(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = torch.cat([torch.unsqueeze(x, dim=0) for x in batch], dim=0)
    x0 = apply_transforms(batch)
    x1 = apply_transforms(batch)

    return x0, x1

if __name__ == '__main__':
    dataset = ShortVideoDataset('video_short', transform=T.Compose([
    T.ToTensor(),
    T.CenterCrop(size=720)
]))
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    batch = next(iter(dataloader))

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111, 
        nrows_ncols=(5, 5),
        axes_pad=0.1
    )

    for ax, im in zip(grid, torch.cat(tuple(apply_transforms(batch) for _ in range(5)), dim=0)):
        ax.imshow(im.swapaxes(0,-1).swapaxes(0,1))

    plt.show()