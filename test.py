from __future__ import print_function
import random
import torch.nn as nn
import torch.utils.data
from model import Generator
import torchvision.utils as vutils
from dataset import plot

model_path = 'epochs/epoch_G_300.pth'

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# 图片的分辨率，默认为64x64。如果使用了其它尺寸，需要更改判别器与生成器的结构
image_size = 64
# 生成器的输入zz 的维度
nz = 100
b_size = 64
class_num = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

netG = Generator(64)
netG.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
netG.to(device)

noise = torch.randn(b_size, nz, 1, 1, device=device)
# Generate fake image batch with G
imgs = netG(noise)

for i, img in enumerate(imgs):
    plot(img[0].detach().cpu().numpy(), './test_results/test_%d.jpg'%i, False)



