import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from dataloader import R2D2Dataset, collate_fn
from loss import CosimLoss, PeakyLoss, APLoss
from R2D2 import R2D2

parser = argparse.ArgumentParser(description='R2D2')
parser.add_argument('--dataPath', type=str, default='/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data')
parser.add_argument('--savePath', type=str)
parser.add_argument('--resume', type=str, defatult='')
parser.add_argument('--multigpu', type=bool, default=False)

root_dir = '/home/jhyeup/myr2d2/r2d2'

batch_size = 8
lr = 1e-4
weight_decay = 5e-4
threads = 8
start_epoch = 0
nEpochs = 25

def train(epoch) :

    epoch_loss = 0
    
    dataloader = DataLoader(dataset, num_workers=threads, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    nBatches = (len(dataset) + batch_size - 1) // batch_size
    r2d2.train()
    for iteration, (source_images, transform_images, metas) in enumerate(dataloader, 1) :

        loss = 0
        for b in range(len(source_images)) :
            image1 = source_images[b].to(device)
            image2 = transform_images[b].to(device)
            meta = metas[b]
            meta['grid'] = meta['grid'].to(device)
            if 'mask' in meta :
                meta['mask'] = meta['mask'].to(device)

            descriptors, reliabilities, repeatabilities = r2d2([image1, image2])
            loss1 = cosimloss.forward_one(repeatabilities[0][0], repeatabilities[1][0], meta)
            loss2 = (peakyloss.forward_one(repeatabilities[0][0]) + peakyloss.forward_one(repeatabilities[1][0]) )/2
            loss3 = aploss.forward_one(descriptors[0][0], descriptors[1][0], reliabilities[0][0], meta)
            loss += loss1 + loss2 + loss3

        loss = loss/batch_size
        loss.to(device)
        loss.backward()
        optimizer.step()
        del input, descriptors, reliabilities, repeatabilities

        batch_loss = loss.item()
        epoch_loss += batch_loss

    del dataloader, loss
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    epoch_loss /= nBatches

    print("===> Epoch {} Complete : Avg. Loss: {:.4f}".format(epoch, epoch_loss), flush=True)
    writer.add_scalar('Train/AvgLoss', epoch_loss, epoch)

def save_checkpoint(savePath, state, filename='checkpoint.pth.tar') :
    model_out_path = os.path.join(savePath, filename)
    torch.save(state, model_out_path)

if __name__ == "__main__":

    args = parser.parse_args()

    if not torch.cuda.is_available() :
        raise Exception("GPU failed")
    torch.cuda.empty_cache()

    device = torch.device('cuda')

    print("===> Loading Dataset")
    dataset = R2D2Dataset(args.dataPath)
    print("Done Load")

    print("===> Building model")
    r2d2 = R2D2()

    if args.multigpu :
        r2d2 = nn.DataParallel(r2d2)
        batch_size = batch_size * 7

    optimizer = optim.Adam(filter(lambda p : p.requires_grad, r2d2.parameters()), lr=lr, weight_decay=weight_decay)

    if args.resume :
        print("===> Loading model")
        checkpoint = torch.load(args.resume)
        r2d2.load_state_dict(checkpoint['state_dict'])
        r2d2.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print("Done Loading")

    else :
        r2d2.to(device)

    #criterion here
    cosimloss = CosimLoss().to(device)
    peakyloss = PeakyLoss().to(device)
    aploss = APLoss().to(device)

    print("Done Build")

    print("===> Training")

    if not os.path.exists(os.path.join(root_dir, 'log')) :
        os.makedirs(os.path.join(root_dir, 'log'))

    writer = SummaryWriter(log_dir=os.path.join(root_dir, 'log', datetime.now().strftime('%b%d_%H-%M-%S')))
    logdir = writer.file_writer.get_logdir()
    savePath = os.path.join(logdir, 'checkpoints')
    os.makedirs(savePath)

    for epoch in range(start_epoch+1, nEpochs+1) :
        train(epoch)

        save_checkpoint(savePath, {
            'epoch' : epoch,
            'state_dict' : r2d2.state_dict(),
            'optimizer' : optimizer.state_dict()
        })

    writer.close()


    

