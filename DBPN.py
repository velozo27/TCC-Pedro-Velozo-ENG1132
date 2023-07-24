import torch
import torch.nn as nn


class DBPN(nn.Module):
    def __init__(self) -> None:
        super(DBPN, self).__init__()

        # meta-parameters
        self.kernel, self.stride, self.padding = self._get_kernel_data()

        # Initial feature extraction
        # Our final network, D-DBPN, uses conv(3,256) then conv(1, 64) for the initial feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 64, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )

        # Back-projection stages layer
        # For testing, we will use T = 2 (model of type S)
        # Back-projection stages layer
        self.up_block1 = UpProjectionBlock(
            64, self.kernel, self.stride, self.padding)
        self.down_block1 = DownProjectionBlock(
            64, self.kernel, self.stride, self.padding)
        self.up_block2 = UpProjectionBlock(
            64, self.kernel, self.stride, self.padding)

        self.deep_down_block1 = DeepDownProjectionBlock(
            64, self.kernel, self.stride, self.padding, num_stage=2)
        self.deep_up_block1 = DeepUpProjectionBlock(
            64, self.kernel, self.stride, self.padding, num_stage=2)

        # Reconstruction
        # We use conv(1, 1) for the reconstruction.
        self.conv3 = nn.Conv2d(192, 3, (3, 3), (1, 1), (1, 1))

    def _get_kernel_data(self):
        # 4× enlargement use 8 × 8 convolutional layer with four striding and two padding
        kernel = 8
        stride = 4
        padding = 2
        return kernel, stride, padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial feature extraction
        x = self.conv1(x)
        x = self.conv2(x)

        # Back-projection stages layer
        # For testing, we will use T = 2 (model of type S)
        h1 = self.up_block1(x)
        l1 = self.down_block1(h1)
        h2 = self.up_block2(l1)

        concat_h = torch.cat((h2, h1), 1)
        l = self.deep_down_block1(concat_h)

        concat_l = torch.cat((l, l1), 1)
        h = self.deep_up_block1(concat_l)

        # Reconstruction
        concat_h = torch.cat((h, concat_h),1)
        x = self.conv3(concat_h)


        return x        


class UpProjectionBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int,
            stride: int,
            padding: int) -> None:
        super(UpProjectionBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size,
                               kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size,
                                           kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size,
                               kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.conv1(x)
        l0 = self.conv2(h0)
        h1 = self.conv3(l0 - x)
        return h1 + h0


class DeepUpProjectionBlock(nn.Module):
    """ 
    This block is different from the UpProjectionBlock in that it has as the input, the concatenation of the output of the previous block and the output of the initial feature extraction block.
    """
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, num_stage: int) -> None:
        super(DeepUpProjectionBlock, self).__init__()
        self.deep_conv = nn.Sequential(
            nn.Conv2d(channels * num_stage, channels, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size,
                               kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size),
                      (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size,
                               kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_conv = self.deep_conv(x)
        conv1 = self.conv1(deep_conv)
        conv2 = self.conv2(conv1)
        conv2 = torch.sub(conv2, deep_conv)
        conv3 = self.conv3(conv2)

        out = torch.add(conv3, conv1)

        return out


class DownProjectionBlock(nn.Module):
    def __init__(
            self,
            channels: int,
            kernel_size: int,
            stride: int,
            padding: int) -> None:
        super(DownProjectionBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size,
                                           kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size,
                               kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size,
                                           kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l0 = self.conv1(x)
        h0 = self.conv2(l0)
        l1 = self.conv3(h0 - x)
        return l1 + l0


class DeepDownProjectionBlock(nn.Module):
    """ 
    This block is different from the DownProjectionBlock in that it has as the input, the concatenation of the output of the previous block and the output of the initial feature extraction block.
    """
    def __init__(self, channels: int, kernel_size: int, stride: int, padding: int, num_stage: int) -> None:
        super(DeepDownProjectionBlock, self).__init__()
        self.deep_conv = nn.Sequential(
            nn.Conv2d(channels * num_stage, channels, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size),
                      (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (kernel_size,
                               kernel_size), (stride, stride), (padding, padding)),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, kernel_size),
                      (stride, stride), (padding, padding)),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        deep_conv = self.deep_conv(x)
        conv1 = self.conv1(deep_conv)
        conv2 = self.conv2(conv1)
        conv2 = torch.sub(conv2, deep_conv)
        conv3 = self.conv3(conv2)

        out = torch.add(conv3, conv1)

        return out



if __name__ == "__main__":
    model = DBPN()

    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    print(output.shape)