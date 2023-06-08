import logging

import torch
from torch import nn


class ConvNorm(nn.Module):
    """
    Simple union of convolutional layer and batch normalization
    """
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, div2: bool = False, padding: int = 1):
        super(ConvNorm, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=2 if div2 else 1,
            padding=padding
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.norm(self.conv(x))


class BaseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, div2: bool = False):
        """
        Base structural component for resnet18 model

        :param in_channels: input will be with shape (batch_size, in_channels, H, W)
        :param out_channels: output will be with shape (batch_size, out_channels, H, W)
        """
        super(BaseLayer, self).__init__()

        self.layer1 = ConvNorm(in_channels, out_channels, div2=div2)
        self.relu = nn.ReLU()
        self.layer2 = ConvNorm(out_channels, out_channels)

    def forward(self, x: torch.Tensor, do_warning=True) -> torch.Tensor:
        out: torch.Tensor = self.layer2(self.relu(self.layer1(x)))
        if x.shape == out.shape:
            return self.relu(out + x)

        if do_warning:
            logging.warning("Can't apply residual step because shapes are different")
        return out


class BaseLayerWithDownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Base resnet layer for cases when image shape was changed

        :param in_channels: input will be with shape (batch_size, in_channels, H, W)
        :param out_channels: output will be with shape (batch_size, out_channels, H, W)
        """
        super(BaseLayerWithDownSample, self).__init__()

        self.main = BaseLayer(in_channels, out_channels, div2=True)
        self.down_sample = ConvNorm(in_channels, out_channels, kernel=1, div2=True, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x, do_warning=False)
        out2 = self.down_sample(x)
        return self.relu(out + out2)
