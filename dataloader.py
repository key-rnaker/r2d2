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

class R2D2Dataset(data.Dataset) :
    def __init__(self, root_dir) :
        super().__init__()

        self.root_dir = root_dir
        self.style_images = os.listdir(os.path.join(self.root_dir, "style_transfer"))
        self.optical_images = os.listdir(os.path.join(self.root_dir, "optical_flow", "flow"))
        self.total_images = self.style_images + self.optical_images
        self.DBidx = np.arange(len(self.total_images))
        np.random.shuffle(self.DBidx)
        self.input_transform = input_transform()

    def get_optical_image_pair(self, image_name) :

        flow_image = Image.open(os.path.join(self.root_dir, 'optical_flow', 'flow', image_name))
        mask_image = Image.open(os.path.join(self.root_dir, 'optical_flow', 'mask', image_name))
        
        flow_array = np.asarray(flow_image)
        mask_array = np.asarray(mask_image)

        source_image_name = image_name.split('_')[0] + '.jpg'
        transform_image_name = image_name.split('_')[1].split('.')[0] + '.jpg'

        source_image = Image.open(os.path.join(self.root_dir, 'aachen', 'images_upright', 'db', source_image))
        transform_image = Image.open(os.path.join(self.root_dir, 'aachen', 'images_upright', 'db', transform_image_name))

        W, H = source_image.size

        mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1,2,0).astype(np.float32)
        aflow = flow_array + mgrid

        meta = {}
        meta['aflow'] = aflow
        meta['mask'] = mask_array

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

        meta = {}
        meta['aflow'] = mgrid * (sx, sy)

        return source_image, transform_image, meta

    def get_image_pair(self, image_name) :

        if image_name.split('.')[-1] == 'png' :
            return self.get_optical_image_pair(image_name)

        else :
            return self.get_style_image_pair(image_name)

    def __getitem__(self, index) :
        
        idx = self.DBidx[index]

        source_image, transform_image, meta = self.get_image_pair(self.total_images[idx])

        source_image = self.input_transform(source_image)
        transform_image = self.input_transform(transform_image)

        return source_image, transform_image, meta

    def __len__(self) :
        return len(self.total_images)


