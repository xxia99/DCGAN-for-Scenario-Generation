import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(w):
    classname = w.__class__.__name__
    if (type(w) == nn.ConvTranspose2d or type(w) == nn.Conv2d):
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif (type(w) == nn.BatchNorm2d):
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)
    elif (type(w) == nn.Linear):
        nn.init.normal_(w.weight.data, 0.0, 0.02)

# G(z)
class Generator(nn.Module):
    def __init__(self, ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(  # 一个序列容器，将模块按照次序添加到这个容器里面组成一个model
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            # c是输入特征图的数目，也就是channel数目，对每个特征图上的点进行减均值除方差的操作（均值和方差是每个mini-batch内的对应feature层的均值和方差）
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, (2, 1), (2, 2), bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*4) x 8 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, (2, 1), (1, 2), bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf*2) x 16 x 4
            nn.ConvTranspose2d(ngf * 2, ngf, 4, (2, 1), (1, 1), bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, True),
            # state size. (ngf) x 32 x 4
            nn.ConvTranspose2d(ngf, 1, 4, (2, 1), (1, 1), bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 4
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 48 x 4
            nn.Conv2d(1, ndf, (4, 3), (2, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 24 x 4
            nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 2
            nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 1
            nn.Conv2d(ndf * 4, ndf * 8, (4, 1), (2 ,1), (1, 0), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 1
            nn.Conv2d(ndf * 8, 1, (3, 1), (1, 1), (0,0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class WDiscriminator(nn.Module):
    def __init__(self, ndf):
        super(WDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 48 x 4
            nn.Conv2d(1, ndf, (4, 3), (2, 1), (1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 24 x 4
            nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 2
            nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 1
            nn.Conv2d(ndf * 4, ndf * 8, (4, 1), (2 ,1), (1, 0), bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 3 x 1
            nn.Conv2d(ndf * 8, 1, (3, 1), (1, 1), (0,0), bias=False)
        )

    def forward(self, input):
        return self.main(input)
