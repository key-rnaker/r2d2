from dataloader import *
import torch
from torchvision import transforms
from R2D2 import *

dataset = R2D2Dataset(root_dir='/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/')
# len(dataset) = 8406

image1, image2, meta1 = dataset.__getitem__(2)

r2d2 = R2D2()
r2d2.to(torch.device('cuda'))
#input = torch.cat([image1, image2])
#input = input.to(torch.device('cuda'))
image1 = image1.to(torch.device('cuda'))
d, rl, rp = r2d2.forward_one(image1)