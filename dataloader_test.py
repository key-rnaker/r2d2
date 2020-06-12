from dataloader import *
import torch
from torchvision import transforms

dataset = R2D2Dataset(root_dir='/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/')
# len(dataset) = 8406

source_image, transform_image, meta = dataset.__getitem__(2)

def tran(image1, image2, meta) :
    #image1 = transforms.Resize((image1.size[1]//2, image1.size[0]//2))(image1)
    #image2 = transforms.Resize((image2.size[1]//2, image2.size[0]//2))(image2)
    tensor1 = transforms.ToTensor()(image1)
    tensor2 = transforms.ToTensor()(image2)

    """
    grid = meta['grid']
    grid = grid.permute(0,3,1,2)
    pool_grid = torch.nn.MaxPool2d(2)(grid)
    meta['grid'] = pool_grid.permute(0,2,3,1)
    print("meta['grid'] : ",meta['grid'].shape)

    if 'mask' in meta :
        mask = torch.from_numpy(meta['mask'].astype(np.float))
        mask = mask.unsqueeze(0).unsqueeze(0)
        pool_mask = torch.nn.MaxPool2d(2)(mask)
        meta['mask'] = pool_mask.squeeze(0).squeeze(0)
        mymask = meta['mask'].numpy().astype(np.float) - 1
        meta['grid'][0,:,:,0] += mymask
        meta['grid'][0,:,:,1] += mymask
    """
    tensor1f2 = torch.nn.functional.grid_sample(tensor2.unsqueeze(0), meta['grid']).squeeze(0)

    """
    if 'mask' in meta :
        for i in range(meta['mask'].shape[0]) :
            for j in range(meta['mask'].shape[1]) :
                if meta['mask'][i,j] == 0 :
                    tensor1f2[:,i,j] = torch.zeros([3])
                else :
                    continue
    """
    image1f2 = transforms.ToPILImage()(tensor1f2)
    image1f2.show()
    print(image1f2.size)

def tran2(number) :
    a, b, c = dataset.__getitem__(number)
    tran(a,b,c)
    print(dataset.total_images[number])