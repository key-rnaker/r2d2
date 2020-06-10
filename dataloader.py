import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image

def input_transform():
    # pre trained VGG16 model expects input images normalized
    # mean and std of ImageNet
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])

def collate_fn(batch) :

    batch = list(filter(lambda x : x is not None, batch))
    if len(batch) == 0 : return None, None, None

    source_image, transform_image, meta = zip(*batch)
    """
    source_image : tuple (batch_size, tensor(3, h, w))
    transform_image : tuple (batch_size, tensor(3, h, w))
    meta : tuple (batch_size, dict['aflow', 'mask'])
    """

    source_images = data.dataloader.default_collate(source_image)
    transform_images = data.dataloader.default_collate(transform_image)
    metas = data.dataloader.default_collate(meta)

    return source_images, transform_images, metas


class R2D2Dataset(data.Dataset) :
    def __init__(self, root_dir) :
        super().__init__()

        self.root_dir = root_dir
        self.style_images = os.listdir(os.path.join(self.root_dir, "style_transfer"))
        self.optical_images = os.listdir(os.path.join(self.root_dir, "optical_flow", "flow"))
        self.total_images = self.style_images + self.optical_images
        self.input_transform = input_transform()

    def flowimage_to_aflow(self, flow_image) :
        flow = np.asarray(flow_image).view(np.int16)
        flow = np.float32(flow) / 16
        W, H = flow_image.size
        mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1,2,0).astype(np.float32)
        aflow = flow + mgrid
        return aflow

    def aflow_to_grid(self, aflow) :
        H, W = aflow.shape[:2]
        grid = torch.from_numpy(aflow).unsqueeze(0)
        grid[:,:,:,0] *= 2/(W-1)
        grid[:,:,:,1] *= 2/(H-1)
        grid -= 1
        grid[torch.isnan(grid)] = 9e9
        return grid

    def get_label(self, image1_size) :
        W, H = image1_size
        label = torch.zeros([H,W,H,W])
        for i in range(H) :
            for j in range(W) :
                label = torch.zeros([H, W])
                label[i,j][max(0, i-4) : min(H1, i+5), max(0, j-4) : min(W1, j+5)] = 1

        return label

    def get_optical_image_pair(self, image_name) :

        flow_image = Image.open(os.path.join(self.root_dir, 'optical_flow', 'flow', image_name))
        mask_image = Image.open(os.path.join(self.root_dir, 'optical_flow', 'mask', image_name))
        
        mask_array = np.asarray(mask_image)

        source_image_name = image_name.split('_')[0] + '.jpg'
        transform_image_name = image_name.split('_')[1].split('.')[0] + '.jpg'

        source_image = Image.open(os.path.join(self.root_dir, 'aachen', 'images_upright', 'db', source_image_name))
        transform_image = Image.open(os.path.join(self.root_dir, 'aachen', 'images_upright', 'db', transform_image_name))

        W, H = source_image.size

        aflow = self.flowimage_to_aflow(flow_image)
        grid = self.aflow_to_grid(aflow)

        meta = {}
        meta['grid'] = grid
        meta['mask'] = mask_array
        meta['label'] = self.get_label(source_image.size)

        return source_image, transform_image, meta

    def get_style_image_pair(self, image_name) :

        source_image_name = image_name.split('.')[0] + '.' + image_name.split('.')[1]
        source_image = Image.open(os.path.join(root_dir, 'aachen', 'images_upright', 'db', source_image_name))
        source_image = self.input_transform(source_image)

        transform_image = Image.open(os.path.join(root_dir, 'style_transfer', image_name))
        transform_image = self.input_transform(transform_image)

        W, H = source_image.size
        sx = transform_image.size[0] / float(W)
        sy = transform_image.size[1] / float(H)

        mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1,2,0).astype(np.float32)
        aflow = mgrid * (sx, sy)
        grid = self.aflow_to_grid(aflow)

        meta = {}
        meta['grid'] = grid
        meta['label'] = self.get_label(source_image.size)

        return source_image, transform_image, meta

    def get_image_pair(self, image_name) :

        if image_name.split('.')[-1] == 'png' :
            return self.get_optical_image_pair(image_name)

        else :
            return self.get_style_image_pair(image_name)

    def __getitem__(self, index) :
        
        source_image, transform_image, meta = self.get_image_pair(self.total_images[index])

        source_image = self.input_transform(source_image)
        transform_image = self.input_transform(transform_image)

        return source_image, transform_image, meta

    def __len__(self) :
        return len(self.total_images)


