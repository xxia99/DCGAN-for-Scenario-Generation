from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

from model import Generator, WDiscriminator, weights_init
from dataset import MyDataset, plot

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

checkpoint_G_path = None
checkpoint_D_path = None
batch_size = 128
lr = 0.0002
num_epochs = 500
class_num = 2
image_size = (64, 64)
n_critic = 5
clip_value = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model and load pretrained-model
netG = Generator(64)
netD = WDiscriminator(64)
netG.apply(weights_init)
netD.apply(weights_init)

if checkpoint_G_path is None:
    # netG.init_weight()
    pass
else:
    netG.load(checkpoint_G_path)
if checkpoint_D_path is None:
    # netD.init_weight()
    pass
else:
    netD.load(checkpoint_D_path)

netG = netG.to(device)
netD = netD.to(device)

train_dataset = MyDataset('./Datasets')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_iter_per_epoch = len(train_dataset) // batch_size + 1

optimG = torch.optim.RMSprop(netG.parameters(), lr=lr)
optimD = torch.optim.RMSprop(netD.parameters(), lr=lr)

ce_loss = nn.BCELoss()

summary = SummaryWriter()

for epoch in range(1, 1+num_epochs):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'D_loss': 0, 'G_loss': 0, 'D_score': 0, 'G_score': 0}
    for data in train_bar:
        running_results['batch_sizes'] += 1

        # Prepare data
        real_data = data.to(device)
        noise = torch.randn((real_data.shape[0], 100)).view(-1, 100, 1, 1).to(device)

        fake_data = netG(noise)
        for i, img in enumerate(fake_data[:2]):
            plot(img[0].detach().cpu().numpy(), './results_w/w_%d_%d.jpg'%(epoch-1, i), False)
        ones = torch.ones(real_data.shape[0]).to(device)
        zeros = torch.zeros(real_data.shape[0]).to(device)

        # Train D
        real_D = netD(real_data).reshape(-1)
        fake_D = netD(fake_data.detach()).reshape(-1)
        #D_loss = ce_loss(real_D, ones) + ce_loss(fake_D, zeros)
        D_loss = -real_D.mean() + fake_D.mean()

        optimD.zero_grad()
        D_loss.backward()
        optimD.step()

        # Clip weights of discriminator
        for p in netD.parameters():
            p.data.clamp_(-clip_value, clip_value)

        running_results['D_loss'] += D_loss.mean().item() * batch_size
        running_results['D_score'] = real_D.mean().item()
        running_results['G_score'] = fake_D.mean().item()
        score = {
            "realD":  real_D.mean().item(),
            "fakeD":  fake_D.mean().item()
        }
        summary.add_scalar(tag="D_loss", scalar_value=D_loss.mean().item(),
                           global_step=running_results['batch_sizes'] + (epoch-1)*num_iter_per_epoch)
        summary.add_scalar(tag="Wasserteion Distance", scalar_value=-D_loss.mean().item(),
                           global_step=running_results['batch_sizes'] + (epoch-1)*num_iter_per_epoch)
        summary.add_scalars("D", score,
                           global_step=running_results['batch_sizes'] + (epoch-1)*num_iter_per_epoch)

        if running_results['batch_sizes'] % n_critic == 0:
            # Train G
            fake_D = netD(fake_data).reshape(-1)
            #G_loss = ce_loss(fake_D, ones)
            G_loss = -fake_D.mean()

            optimG.zero_grad()
            G_loss.backward()
            optimG.step()

            running_results['G_loss'] += D_loss.mean().item()

            summary.add_scalar(tag="G_loss", scalar_value=G_loss.mean().item(),
                           global_step=running_results['batch_sizes'] + (epoch-1)*num_iter_per_epoch)

            train_bar.set_description(
                desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f '
                     'Score_D: %.4f Score_G: %.4f' % (
                epoch, num_epochs,
                running_results['D_loss']/running_results['batch_sizes'],
                running_results['G_loss']/running_results['batch_sizes'],
                running_results['D_score'],
                running_results['G_score'],
                ))

    if epoch % 1 == 0:
        torch.save(netG.state_dict(), 'epochs_w/epoch_w_G_%d.pth' % epoch)
        torch.save(netD.state_dict(), 'epochs_w/epoch_w_D_%d.pth' % epoch)

