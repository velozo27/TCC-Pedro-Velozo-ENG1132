import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self) -> None:
        """
        Initialize SRCNN model by defining its architecture.

        This model implements the Super-Resolution Convolutional Neural Network (SRCNN),
        which is a deep neural network designed for image super-resolution tasks.
        It consists of three main steps: patch extraction and representation, non-linear mapping,
        and image reconstruction.

        Args:
        None.

        Returns:
        None.
        """
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            # 3.1.1 Patch extraction and representation
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=3, padding=0),
            nn.ReLU(),

            # 3.1.2 Non-linear mapping
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=0),
            nn.ReLU(),

            # 3.1.3 Reconstruction
            nn.Conv2d(32, 3, kernel_size=5, padding=0),
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        """
        Forward pass of SRCNN model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        nn.Sequential: Output tensor of shape (batch_size, 3, height, width) after applying
        patch extraction, non-linear mapping, and image reconstruction operations.
        """
        x = self.model(x)
        return x
