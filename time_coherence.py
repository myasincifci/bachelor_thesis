from tqdm import tqdm

import torch
from torch import nn
import torchvision
import torchvision.transforms as T

from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss
from custom_loss import CustomLoss

from time_dataset import ShortVideoDatasetTime

from augmentation import apply_transforms

BATCH_SIZE = 1024


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


def main():

    torch.manual_seed(42)

    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone[0] = nn.Conv2d(1, 64, kernel_size=(
        3, 3), stride=(2, 2), padding=(3, 3), bias=False)
    model = BarlowTwins(backbone)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    dataset = ShortVideoDatasetTime('video_short_half_res', transform=T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Grayscale()
    ]))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
    )

    # criterion = BarlowTwinsLoss(lambda_param=1e-3)
    criterion = CustomLoss(batch_size=BATCH_SIZE, l=1e-3)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, weight_decay=0.001)

    print("Starting Training")
    for epoch in range(50):
        total_loss = 0
        for index, batch in tqdm(dataloader):
            x = batch.to(device)
            z = model(x)
            loss = criterion(z)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

        torch.save(model, 'models/model_time.pth')


if __name__ == '__main__':
    main()
