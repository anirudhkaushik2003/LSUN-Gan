import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import PIL

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 5, padding='same')
        self.bnorm = nn.BatchNorm2d(out_ch)
        # self.pool = nn.MaxPool2d(2, 2)
        # replace pooling with strided convolutions
        self.pool = nn.Conv2d(out_ch, out_ch, 2, stride=2)
        # output size = (input_size - kernel_size + 2*padding)/stride + 1 = (32 - 2 + 2*0)/2 + 1 = 16
        # used leaky relu for stable training
        # used batchnorm for stable training
        # didn't use any fully connected hidden layers
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)

        x = self.pool(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels

        self.conv1 = Block(img_channels, 64) # 32x32x1 -> 16x16x64
        self.conv2 = Block(64, 128) # 16x16x64 -> 8x8x128
        self.conv3 = Block(128, 256) # 8x8x128 -> 4x4x256 
        self.conv4 = Block(256, 512) # 4x4x256 -> 2x2x512
        

        self.out = nn.Conv2d(512, 1, 2 ) # 2x2x512 -> 1x1x1
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.out(x)
        x = self.out_act(x)
        x = nn.Flatten()(x) # 1x1x1 -> 1x1

        
        return x
