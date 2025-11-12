import torch
import torch.nn as nn
import math

class BlurPool2d(nn.modules):
    def __init__(self, kernel_size = 4, stride = 2):
        self.kernel_size = kernel_size
        # This block here of custom kernels is taken almost directly from the MaskGIT implementation
        if self.kernel_size == 3:
            self.filter = [1., 2., 1.]
        elif self.kernel_size == 4:
            self.filter = [1., 3., 3., 1.]
        elif self.kernel_size == 5:
            self.filter = [1., 4., 6., 4., 1.]
        elif self.kernel_size == 6:
            self.filter = [1., 5., 10., 10., 5., 1.]
        elif self.kernel_size == 7:
            self.filter = [1., 6., 15., 20., 15., 6., 1.]
        else:
            raise ValueError('Only filter_size of 3, 4, 5, 6 or 7 is supported.')

        self.filter = torch.tensor(self.filter)
        self.filter = torch.outer(self.filter, self.filter)
        self.filter /= torch.sum(self.filter)
        # F.conv2d takes: weight – filters of shape (out_channels,in_channels / groups,kH,kW)
        filter = torch.reshape(self.filter, [1, 1, self.kernel_size, self.kernel_size])
    
    def forward(self, x):
        channel_depth = x.shape[1] # input is n x c x h x w so idx 1 is c
        depthwise_filter = torch.tile(filter, [channel_depth, 1, 1, 1])
        x = torch.nn.functional.conv2d(x,
                                weight = depthwise_filter,
                                stride = self.stride,
                                groups = channel_depth,
                                padding = 1)
        return x

class ResBlock(nn.module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # MaskGIT implementation appears to only resize on the second conv, so I will do the same here too. I should figure out why that is.
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.increase_channels = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.activation(x)
        x = BlurPool2d(x)
        x = self.conv2(x)
        x = self.activation(x)

        if (self.in_channels != self.out_channels):
            residual = self.increase_channels(residual)

        x = (x + residual) / math.sqrt(2)