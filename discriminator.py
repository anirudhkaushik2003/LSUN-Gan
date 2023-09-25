import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import PIL

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.bnorm = nn.BatchNorm2d(out_ch)
        # self.pool = nn.MaxPool2d(2, 2)
        # replace pooling with strided convolutions
        # self.pool = nn.Conv2d(out_ch, out_ch, 2, stride=2)
        # output size = (input_size - kernel_size + 2*padding)/stride + 1 = (32 - 2 + 2*0)/2 + 1 = 16
        # used leaky relu for stable training
        # used batchnorm for stable training
        # didn't use any fully connected hidden layers
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, IMG_SIZE=128):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = IMG_SIZE
        self.out_ch = self.img_size*8

        self.conv1 = Block(img_channels, self.out_ch//16) # 128x128x3 -> 64x64x64
        self.conv2 = Block(self.out_ch//16, self.out_ch//8) # 64x64x64 -> 32x32x128
        self.conv3 = Block(self.out_ch//8, self.out_ch//4) # 32x32x128 -> 16x16x256 
        self.conv4 = Block(self.out_ch//4, self.out_ch//2) # 16x16x256 -> 8x8x512
        self.conv5 = Block(self.out_ch//2, self.out_ch) # 8x8x512 -> 4x4x1024

        

        self.out = nn.Conv2d(self.out_ch, 1, 4, 1, 0, bias=False ) # 4x4x1024 -> 1x1x1
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.out(x)
        x = self.out_act(x)
        
        return x
