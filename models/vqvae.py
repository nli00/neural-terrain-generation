import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLu()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.increase_channels = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 1)
    

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # If we change the number of channels, we need to adjust the dimensions of the residual so the following add doesnt break
        if self.in_channels != self.out_channels:
            residual = self.increase_channels(residual)

        x += residual

        x = self.relu(x)
        return x

class Encoder(nn.module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        # This will all go in a config file later
        num_res_blocks = 2
        data_channels = 1
        filters = 128
        channel_multipliers = [1, 1, 2, 2, 4]
        latent_dim = 256
        # ------

        first_conv = nn.Conv2d(data_channels, filters, kernel_size = 3, stride = 1, padding = 1)
        layers = [first_conv]


        # So I could either have the ResBlock multiply the channels or I could have the downsample layer do it
        # I'm going with the latter because it should be less computationally expensive
        # But the former could potentially be more expressive as the second ResBlock would get to work at a deeper channel depth with the full resolution
        num_blocks = len(channel_multipliers)
        for i in range(num_blocks - 1):
            in_channels = filters * channel_multipliers[i]

            for _ in range(num_res_blocks):
                layers.append(ResBlock(in_channels, in_channels))

            if i < num_blocks - 1:
                # Halves the resolution and multiplies the channels according to the channel_multipliers
                out_channels = filters * channel_multipliers[i + 1]
                downsample_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 2)
                layers.append(downsample_layer)

        for _ in range(num_res_blocks):
            final_channels = filters * channel_multipliers[-1]
            layers.append(ResBlock(final_channels, final_channels))

        layers.append(nn.BatchNorm2D(final_channels))
        layers.append(nn.ReLu())
        layers.append(nn.Conv2d(final_channels, latent_dim, kernel_size = 1, stride = 1, padding = 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, codebook_dim):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x):
        b, _, h, w = x.shape # n x c x h x w
        z_e = x.permute(0, 2, 3, 1) # n x h x w x c
        z_e_flat = z_e.reshape(-1, self.codebook_dim) # (n * h * w) x c

        # Squared euclidean distance is a^2 + b^2 - 2ab for each dimension of the space
        distances = torch.sum(z_e_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.codebook.weight**2, dim=1) - \
            2 * torch.matmul(z_e_flat, self.codebook.weight.t())
        
        codebook_indices = torch.argmin(distances, dim = 1)

        z_q = self.codebook(codebook_indices) # (n * h * w) x c 
        z_q = z_q.reshape(b, h, w, self.codebook_dim) # n x h x w x c

        loss = F.mse_loss(z_e.detach(), z_q) + F.mse_loss(z_q.detach(), z_e)

        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.permute(0, 3, 1, 2) # n x c x h x w

        return z_q, loss, codebook_indices