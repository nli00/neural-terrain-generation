import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.module):
    def __init__(self, feature_layers=[]):
        super().__init__()

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        