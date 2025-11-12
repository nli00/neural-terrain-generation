import torch
import torch.nn as nn
import torch.nn.functional as F

# Now that I think about it, I really should just infer the in channels from the data shape. 
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.increase_channels = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
    

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

class UpsampleLayer(nn.Module):
    def __init__(self, channels):
        super(UpsampleLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        # print(x.shape)
        x = F.interpolate(x, scale_factor = 2.0, mode = 'nearest')
        x = self.conv(x)
        # print(x.shape)
        return x

class DownsampleLayer(nn.Module):
    def __init__(self, channels):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 4, stride = 2, padding = 1)
    
    def forward(self, x):
        # pad = (1, 2, 1, 2)
        # x = F.pad(x, pad, mode="constant", value = 0)
        x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        input_channels = config['data_channels']
        num_res_blocks = config['vqvae']['num_res_blocks']
        filters = config['vqvae']['filters']
        channel_multipliers = config['vqvae']['channel_multipliers']
        latent_dim = config['vqvae']['latent_dim']

        first_conv = nn.Conv2d(input_channels, filters, kernel_size = 3, stride = 1, padding = 1)
        layers = [first_conv]

        # So I could either have the ResBlock multiply the channels or I could have the downsample layer do it
        # // I'm going with the latter because it should be less computationally expensive
        # // But the former could potentially be more expressive as the second ResBlock would get to work at a deeper channel depth with the full resolution
        # Never mind, im having the ResBlock multiply the channels because the maskGIT implementation does that
        # Another thing to note is that I'm not really sure if the downsample layer goes before or after each residual block. I have it after.
        num_blocks = len(channel_multipliers)
        in_channels = filters * channel_multipliers[0]
        for i in range(num_blocks):
            out_channels = filters * channel_multipliers[i]

            layers.append(ResBlock(in_channels, out_channels))
            for _ in range(num_res_blocks - 1):
                layers.append(ResBlock(out_channels, out_channels))

            if i < num_blocks - 1:
                # // Halves the resolution and multiplies the channels according to the channel_multipliers
                # out_channels = filters * channel_multipliers[i + 1]
                # downsample_layer = nn.Conv2d(out_channels, out_channels, kernel_size = 4, stride = 2, padding = 2)
                downsample_layer = DownsampleLayer(out_channels)
                layers.append(downsample_layer)

            in_channels = out_channels
 
        for _ in range(num_res_blocks):
            final_channels = filters * channel_multipliers[-1]
            layers.append(ResBlock(final_channels, final_channels))

        layers.append(nn.BatchNorm2d(final_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(final_channels, latent_dim, kernel_size = 1, stride = 1, padding = 0)) # Squish hidden dim back down to latent dim

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

        return z_q, codebook_indices, loss
    
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        output_channels = config['data_channels']
        num_res_blocks = config['vqvae']['num_res_blocks']
        filters = config['vqvae']['filters']
        channel_multipliers = config['vqvae']['channel_multipliers']
        latent_dim = config['vqvae']['latent_dim']

        # This could totally be a 1x1 convolution to mirror the structure of the last convolution of the encoder, but I'm following what 
        # maskGIT is doing here and starting with a 3x3
        first_conv = nn.Conv2d(latent_dim, filters * channel_multipliers[-1], kernel_size = 3, stride = 1, padding = 1)
        layers = [first_conv]

        for _ in range(num_res_blocks):
            initial_channels = filters * channel_multipliers[-1]
            layers.append(ResBlock(initial_channels, initial_channels))

        num_blocks = len(channel_multipliers)
        in_channels = filters * channel_multipliers[-1]
        for i in reversed(range(num_blocks)):
            out_channels = filters * channel_multipliers[i]

            layers.append(ResBlock(in_channels, out_channels))
            for _ in range(num_res_blocks - 1):
                layers.append(ResBlock(out_channels, out_channels))

            if i > 0:
                # out_channels = filters * channel_multipliers[i - 1]
                upsample_layer = UpsampleLayer(out_channels)
                layers.append(upsample_layer)
            
            in_channels = out_channels

        final_channels = filters * channel_multipliers[0]
        layers.append(nn.BatchNorm2d(final_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(final_channels, output_channels, kernel_size = 3, stride = 1, padding = 1)) # Squish back down to the channels of the image we are trying to produce

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(config)
        self.vq_layer = VectorQuantizer(codebook_size = config['vqvae']['codebook_size'], codebook_dim = config['vqvae']['latent_dim'])
        self.decoder = Decoder(config)

    
    def forward(self, batch):
        encoded_images = self.encoder(batch)
        quantized_vectors, codebook_indices, codebook_loss = self.vq_layer(encoded_images)
        decoded_images = self.decoder(quantized_vectors)

        return decoded_images, codebook_indices, codebook_loss
    
    def load_checkpoints(self, path):
        self.load_state_dict(torch.load(path)['model_state_dict'])