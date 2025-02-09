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
    



class CustomResnet18(nn.Module):
    def __init__(self, in_channels, num_classes=3):
        super(CustomResnet18, self).__init__()
        resnet = models.resnet18(weights='DEFAULT')  # Not using pre-trained weights
        # Change the first convolutional layer for MNIST
        # resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Adjust the last fully connected layer for MNIST
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)


class CNN(nn.Module):
    
    def __init__(self):
        super(CNN,self).__init__()
        # First we'll define our layers
        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        
        self.maxpool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256 * 2 * 2,512)
        self.fc2 = nn.Linear(512,3)
        
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm2(x)
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.batchnorm3(x)
        x = self.maxpool(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x,dim=1)
        return x
        
        