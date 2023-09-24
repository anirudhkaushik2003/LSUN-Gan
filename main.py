import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

import os
from generator import Generator
from discriminator import Discriminator
from utils import weights_init, create_checkpoint, restart_last_checkpoint

import torchvision.utils as vutils

BATCH_SIZE = 64
IMG_SIZE = 32

data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))

])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train = torchvision.datasets.MNIST(root='/ssd_scratch/cvit/anirudhkaushik/datasets/', train=True, download=False, transform=data_transforms)
test = torchvision.datasets.MNIST(root='/ssd_scratch/cvit/anirudhkaushik/datasets/', train=False, download=False, transform=data_transforms)

data_loader = DataLoader(torch.utils.data.ConcatDataset([train, test]), batch_size=BATCH_SIZE, shuffle=True)


criterion = nn.BCELoss()

modelG = Generator(IMG_SIZE)
modelD = Discriminator()

modelG = modelG.to(device)
modelD = modelD.to(device)

modelG.apply(weights_init)
modelD.apply(weights_init)

fixed_noise = torch.randn(BATCH_SIZE, 100, 1, 1, device='cuda')
real = 1.0
fake = 0.0
learning_rate1 = 2e-4
learning_rate2 = 2e-3
optimD = torch.optim.Adam(modelD.parameters(), lr=learning_rate2, betas=(0.5, 0.999))
optimG = torch.optim.Adam(modelG.parameters(), lr=learning_rate1, betas=(0.5, 0.999))

num_epochs = 100
save_freq = 1

# check if checkpoint exists and load it
if os.path.exists('/ssd_scratch/cvit/anirudhkaushik/checkpoints/gan_checkpoint_latest.pt'):
    restart_last_checkpoint(modelG, optimG, type="G")
    restart_last_checkpoint(modelD, optimD, type="D")

for epoch in range(num_epochs):

    for step, batch in enumerate(data_loader):


        # Train only D model
        modelD.zero_grad()
        real_images = batch[0].to(device)
        real_labels = torch.full((batch[0].shape[0] ,), real, device=device)

        output = modelD(real_images).view(-1)
        lossD_real = criterion(output, real_labels)
        lossD_real.backward()
        D_x = output.mean().item()


        noise = torch.randn(batch[0].shape[0] , 100, 1, 1, device=device) # use gaussian noise instead of uniform
        fake_images = modelG(noise)
        fake_labels = torch.full((batch[0].shape[0] ,), fake, device=device)

        output = modelD(fake_images.detach()).view(-1)
        lossD_fake = criterion(output, fake_labels)

        lossD_fake.backward()
        D_G_z1 = output.mean().item()

        lossD = lossD_real + lossD_fake
        optimD.step()

        # Train only G model
        modelG.zero_grad()
        fake_labels.fill_(real) # use value of 1 so Generator tries to produce real images
        output = modelD(fake_images).view(-1)
        lossG = criterion(output, fake_labels)
        lossG.backward()
        D_G_z2 = output.mean().item()

        optimG.step()

        if step%100 == 0:
            # print(f"Epoch: {epoch}, step: {step:03d}, LossD: {lossD.item()}, LossG: {lossG.item()}, D(x): {D_x}, D(G(z)): {D_G_z1:02f }/{D_G_z2:02f}")
            # limit loss to 2 decimal places
            print(f"Epoch: {epoch}, step: {step:03d}, LossD: {lossD.item():.2f}, LossG: {lossG.item():.2f}, D(x): {D_x:.2f}, D(G(z)): {D_G_z1:.2f}/{D_G_z2:.2f}")
    if epoch%save_freq == 0:
        create_checkpoint(modelG, optimG, epoch, lossG.item(), type="G")
        create_checkpoint(modelD, optimD, epoch, lossD.item(), type="D")


