import numpy as np
import torch
from torchvision import models
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.models import resnet50, ResNet50_Weights


class CustomResNet50(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super(CustomResNet50, self).__init__()
        resnet = models.resnet50(weights="IMAGENET1K_V2")  # Not using pre-trained weights
        # Change the first convolutional layer for MNIST
        # resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust the last fully connected layer for MNIST
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)