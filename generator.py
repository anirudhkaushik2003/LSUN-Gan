import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Block, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4,2,1, bias=False)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        x = self.relu(x)

        return x

class Generator(nn.Module):
    def __init__(self,IMG_SIZE, img_ch=1):
        super(Generator, self).__init__()

        # project and reshape the input
        self.img_size = IMG_SIZE
        self.in_ch = self.img_size*8
        self.img_channels = img_ch


        
        self.project = nn.Sequential(
            nn.ConvTranspose2d(100, self.in_ch, 4,1,0, bias=False),
            nn.BatchNorm2d(self.in_ch),
            nn.LeakyReLU(0.2)
        )
        self.conv1 = Block(self.in_ch, self.in_ch//2)
        self.conv2 = Block(self.in_ch//2, self.in_ch//4)
        self.conv3 = Block(self.in_ch//4, self.in_ch//8)

        # keep output size same as input
        self.out = nn.Conv2d(self.in_ch//8, self.img_channels, 3, padding='same' ) # test with kernel size 3
        self.out_act = nn.Tanh()

    def forward(self, x):
        x = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.out(x)
        x = self.out_act(x)

        return x
    

