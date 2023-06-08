from torch import nn, Tensor


class ConvAndPooling(nn.Module):
    """
    Conv2D: (in_channels, H, W) -> (out_channels, H, W)
    ReLU: (out_channels, H, W) -> (out_channels, H, W)
    MaxPool2D: (out_channels, H, W) -> (out_channels, H // stride[0] + 1, W // stride[1] + 1)
    """
    def __init__(self, in_channels: int, out_channels: int, pool_stride: int | tuple = 2):
        super(ConvAndPooling, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=pool_stride, padding=0)

    def forward(self, x):
        return self.pool(self.relu(self.conv(x)))


class ConvAndBatchNormalization(nn.Module):
    """
    Conv2D: (in_channels, H, W) -> (out_channels, H, W)
    BatchNormalization: (out_channels, H, W) -> (out_channels, H, W)
    ReLU: (out_channels, H, W) -> (out_channels, H, W)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvAndBatchNormalization, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))


class ConvPoolAndConvNorm(nn.Module):
    """
    ConvAndPooling: (in_channels, H, W) -> (out_channels, H // 2, W // 2)
    ConvAndBatchNormalization: (out_channels, H // 2, W // 2) -> (out_channels, H // 2, W // 2)
    """
    def __init__(self, in_channels, out_channels, pool_stride: int | tuple = 2):
        super(ConvPoolAndConvNorm, self).__init__()

        self.conv_pool = ConvAndPooling(in_channels, out_channels, pool_stride=pool_stride)
        self.conv_norm = ConvAndBatchNormalization(out_channels, out_channels)

    def forward(self, x):
        return self.conv_norm(self.conv_pool(x))
