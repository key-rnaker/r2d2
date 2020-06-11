import numpy as np
import torch
from PIL import Image
from torchvision import transforms

flow_image = Image.open('/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/optical_flow/flow/1011_1037.png')
mask_image = Image.open('/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/optical_flow/mask/1011_1037.png')
image1 = Image.open('/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/aachen/images_upright/db/1011.jpg')
image2 = Image.open('/media/jhyeup/5666b044-8f1b-47ad-83f3-d0acf3c6ec52/r2d2data/aachen/images_upright/db/1037.jpg')

def flowimage_to_aflow(flow_image, image1) :
    flow = np.asarray(flow_image).view(np.int16)
    flow = np.float32(flow) / 16
    W,H = image1.size
    mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1,2,0).astype(np.float32)
    aflow = flow + mgrid

    return aflow

def _aflow_to_grid(aflow):
    H, W = aflow.shape[:2]
    grid = torch.from_numpy(aflow).unsqueeze(0)
    grid[:,:,:,0] *= 2/(W-1)
    grid[:,:,:,1] *= 2/(H-1)
    grid -= 1
    grid[torch.isnan(grid)] = 9e9 # invalids
    return grid

aflow = flowimage_to_aflow(flow_image, image1)

image1_array = np.asarray(image1)
image2_array = np.asarray(image2)
mask_array = np.asarray(mask_image)

image1to2_array = np.zeros_like(image2_array)

for i in range(image1.height) :
    for j in range(image1.width) :
        if mask_array[i][j] == 0 :
            continue
        else :
            x, y = aflow[i,j].astype(np.int)
            if 0<= x < image2.width and 0<= y < image2.height :
                image1to2_array[y,x,:] = image1_array[i,j,:]

image1to2 = Image.fromarray(image1to2_array, 'RGB')
image1to2.show(title='image1to2')
grid = _aflow_to_grid(aflow)

image2_tensor = transforms.ToTensor()(image2).unsqueeze(0)
#image1f2_tensor = torch.nn.functional.grid_sample(image2_tensor, grid, mode='bilinear', padding_mode='border')
image1f2_tensor = torch.nn.functional.grid_sample(image2_tensor, grid)

image1f2 = transforms.ToPILImage()(image1f2_tensor.squeeze(0))

image1f2.show(title='image1f2')







