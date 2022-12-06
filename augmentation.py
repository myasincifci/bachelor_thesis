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

def apply_transforms(batch: torch.Tensor) -> torch.Tensor():

    resize = T.Resize(input_size)
    ccrop = T.CenterCrop(size=batch[0].shape[1] * (np.random.rand()*0.3 + 0.7))
    jitter = T.RandomApply([color_jitter], p=cj_prob)
    blur = T.GaussianBlur(kernel_size=3)

    batch_ = T.Grayscale()(batch)

    batch_ = ccrop(batch_)
    batch_ = resize(batch_)

    if np.random.rand() < cj_prob:
        batch_ = jitter(batch_)    
    
    if np.random.rand() < gaussian_blur:
        batch_ = blur(batch_)

    return batch_

def apply_transforms_inf(batch: torch.Tensor) -> torch.Tensor():

    resize = T.Resize(input_size)
    ccrop = T.CenterCrop(size=batch[0].shape[1])

    batch_ = T.Grayscale()(batch)

    batch_ = ccrop(batch_)
    batch_ = resize(batch_)

    return batch_


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

    for ax, im in zip(grid, torch.cat(tuple(apply_transforms_inf(batch) for _ in range(5)), dim=0)):
        ax.imshow(im.swapaxes(0,-1).swapaxes(0,1))

    plt.show()