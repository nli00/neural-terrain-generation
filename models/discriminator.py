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
    def __init__(self, in_channels, out_channels, activation_function):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # MaskGIT implementation appears to only resize on the second conv, so I will do the same here too. I should figure out why that is.
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.activation_function = activation_function

        if in_channels != out_channels:
            self.increase_channels = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.activation_function(x)
        x = BlurPool2d(x)
        x = self.conv2(x)
        x = self.activation_function(x)

        if (self.in_channels != self.out_channels):
            residual = self.increase_channels(residual)

        x = (x + residual) / math.sqrt(2)

# So MaskGIT's authors decided to use a global discriminator, but VQGAN's authors use a patch-based discriminator. I implemented the former, but I should go back and try the other later.
class Discriminator(nn.module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        filters = config['discriminator']['filters']
        channel_multipliers = config['discriminator']['channel_multipliers']
        activation_function = nn.LeakyReLU()

        first_conv = nn.Conv2d(config['data_channels'], filters, kernel_size = 3, stride = 1, padding = 1)
        layers = [first_conv]
        layers.append(activation_function)

        num_blocks = len(channel_multipliers)
        in_channels = filters

        # Basically the logic of this loop should be such that the latent dim is 512 and the latent resolution is 4. 
        # The latent dim should reach 512 by resolution 32 and stop increasing at that point
        # Right now, this is not handled by the code, and instead it is up to the configuration file to do this correctly
        # The reason I'm doing this is because I may want to be able to adjust the latent dim or number of steps or latent resolution at a later point
        # But for now I should stick to what the StyleGAN2 people found to be effective
        for i in range(num_blocks):
            out_channels = filters * channel_multipliers[i]

            layers.append(ResBlock(in_channels, out_channels, activation_function = activation_function))
        
        layers.append(nn.Conv2d(filters*channel_multipliers[-1], filters*channel_multipliers[-1], kernel_size = 3, stride = 1, padding = 1))
        layers.append(activation_function)
        layers.append(nn.Flatten())

        latent_resolution = config['resolution'] / (2 * num_blocks)
        fcl_out = 512 # Using 512 as output dims here just because the MaskGIT implementation did. Should make this configurable
        layers.append(nn.Linear(latent_resolution * config['discriminator']['latent_dim']), fcl_out) 
        layers.append(activation_function)
        layers.append(fcl_out, 1)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)