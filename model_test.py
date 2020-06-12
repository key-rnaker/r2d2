from dataloader import *
import torch
from torchvision import transforms
from R2D2 import *
from loss import *
from torch.utils.data import DataLoader

dataset = R2D2Dataset(root_dir='/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/')
# len(dataset) = 8406

image1, image2, meta = dataset.__getitem__(4)

r2d2 = R2D2()
r2d2.to(torch.device('cuda'))
#input = torch.cat([image1, image2])
#input = input.to(torch.device('cuda'))
"""
image1 = image1.to(torch.device('cuda'))
image2 = image2.to(torch.device('cuda'))
meta['grid'] = meta['grid'].to(torch.device('cuda'))
if 'mask' in meta :
    meta['mask'] = meta['mask'].to(torch.device('cuda'))
d, rl, rp = r2d2.forward([image1, image2])
"""
cosimloss = CosimLoss().to(torch.device('cuda'))

for i in range(10) : 
     image1, image2, meta = dataset.__getitem__(i) 
     image1 = image1.to(torch.device('cuda')) 
     image2 = image2.to(torch.device('cuda')) 
     meta['grid'] = meta['grid'].to(torch.device('cuda')) 
     if 'mask' in meta : 
         meta['mask'] = meta['mask'].to(torch.device('cuda')) 
     d, rl, rp = r2d2.forward([image1, image2]) 
     loss = cosimloss.forward_one(rp[0], rp[1], meta) 
     print(loss)
     del image1, image2, meta, d, rl, rp

"""
peakyloss = PeakyLoss().to(torch.device('cuda'))
aploss = APLoss().to(torch.device('cuda'))

dataloader = DataLoader(dataset, num_workers=8, batch_size=4, shuffle=True, pin_memory=True, collate_fn=collate_fn)

for iteration, (image1s, image2s, metas) in enumerate(dataloader, 1) :
    i1s = image1s
    i2s = image2s
    ms = metas
    break
"""