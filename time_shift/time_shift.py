import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from time_shift_dataset import TimeShiftDataset
from model import TimeShiftModel
from lightly.loss import BarlowTwinsLoss

from tqdm import tqdm

NUM_EPOCHS = 100# 30 # good
LR = 1e-3
BATCH_SIZE = 32
PROXIMITY = 3

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}.')

    transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Grayscale()
    ])

    dataset = TimeShiftDataset('video_short_half_res', transform, proximity=PROXIMITY)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    model = TimeShiftModel().to(device)
    criterion = BarlowTwinsLoss(lambda_param=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)

    for epoch in range(NUM_EPOCHS):
        losses = []
        for image, image_d in tqdm(dataloader):
            image = image.to(device)
            image_d = image_d.to(device)
            
            z0 = model(image)
            z1 = model(image_d)
            loss = criterion(z0, z1)
            losses.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = torch.tensor(losses).mean()
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    torch.save(model, 'models/model_time_shift.pth')

if __name__ == '__main__':
    main()