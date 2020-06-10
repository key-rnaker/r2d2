import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter

from dataloader import R2D2Dataset
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


    return None

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
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, pin_memory=True)
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


    

