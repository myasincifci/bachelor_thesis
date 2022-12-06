from tqdm import tqdm

import torch
from torch import nn
import torchvision
import torchvision.transforms as T

from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss

from short_video_dataset import ShortVideoDataset

from test_augmentations import apply_transforms, collate_fnc

class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
backbone[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
model = BarlowTwins(backbone)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)

dataset = ShortVideoDataset('video_short_half_res', transform=T.Compose([
    T.ToTensor(),
    T.CenterCrop(size=720)
]))
# dataset = LightlyDataset.from_torch_dataset(dataset)

# collate_fn = ImageCollateFunction(input_size=32)
collate_fn = collate_fnc

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    # collate_fn=collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)

criterion = BarlowTwinsLoss(lambda_param=1e-3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

print("Starting Training")
for epoch in range(1000):
    total_loss = 0
    for batch in tqdm(dataloader):
        x0 = apply_transforms(batch).to(device)
        z0 = model(x0)
        del x0
        x1 = apply_transforms(batch).to(device)
        z1 = model(x1)
        del x1
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")