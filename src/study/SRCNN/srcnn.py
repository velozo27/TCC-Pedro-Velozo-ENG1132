import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


class SRCNN(nn.Module):
    def __init__(self) -> None:
        """
        SRCNN model
        """
        super(SRCNN, self).__init__()
        self.conv_relu_conv_stack = nn.Sequential(
            # 3.1.1 Patch extraction and representation
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=9, padding=0),
            nn.ReLU(),

            # 3.1.2 Non-linear mapping
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(),

            # 3.1.3 Reconstruction
            nn.Conv2d(32, 3, kernel_size=5, padding=0),
        )

    def forward(self, x) -> nn.Sequential:
        x = self.conv_relu_conv_stack(x)
        return x

# set path to Set14 dataset
data_path = "./datasets/Set14/"

# define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((33, 33)),   # resize image to 33x33
    transforms.ToTensor()          # convert to tensor
])

# load dataset using ImageFolder
train_dataset = ImageFolder(root=data_path, transform=data_transforms)

# create dataloader to load data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
learning_rate_1 = 1e-4 # for first 2 layers
learning_rate_2 = 1e-5 # for last layer
batch_size = 64
epochs = 5

# Initialize the loss function
# The MSE loss function is evaluated only by the difference between the central pixels of Xi and the network output.
loss_fn = nn.MSELoss()

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Initialize network
model = SRCNN().to(device=device)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_1)

train_loop(model=model,dataloader=train_loader, loss_fn=loss_fn, optimizer=optimizer)