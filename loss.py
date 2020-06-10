import torch
import torch.nn as nn
import torch.nn.functional as F

from aploss import QAPLoss

class CosimLoss(nn.Module) :
    def __init__(self, patch_size=16):
        super().__init__()
        self.N = patch_size

    def extract_pacthes(self, heatmap) :
        """
        Args :
            - heatmap : repeatability heatmap tensor shape of (batch_size, 1, H, W)

        Return :
            - patches : flatten patches tensor shape of (batch_size, (H-(N-1) * W-(N-1)) , N*N)
        """
        patches = nn.Unfold(self.N)
        patches = patches(heatmap).transpose(1,2)
        patches = F.normalize(patches, p=2, dim=2)
        return patches

    def forward(self, repeatability_map, meta) :
        """
        Args :
            - repeatability_map : [image1_repeatability, image2_repeatability]
            - image1_repeatability : repeatability heatmap tensor shape of (batch_size, 1, H, W)
            - image2_repeatability : repeatability heatmap tensor shape of (batch_size, 1, H, W)
            - meta['grid'] : transformation grid (image2 to image1)
        """
        #grid shape = [batch_size, image1_height, image1_width, 2]

        image1_repeatability, image2_repeatability = repeatability_map
        transform_repeatability = F.grid_sample(image2_repeatability, meta['grid'], mode='bilinear', padding_mode='border')
        #transform_repeatability = F.grid_sample(image2_repeatability, grid)

        image1_patches = self.extract_pacthes(image1_repeatability)
        image2_patches = self.extract_pacthes(transform_repeatability)
        cosim = (image1_patches * image2_patches).sum(dim=2)

        return 1 - cosim.mean()

class PeakyLoss(nn.Module) :
    def __init__(self, patch_size=16):
        super().__init__()
        self.N = patch_size
        self.maxpool = nn.MaxPool2d(N+1, stride=1, padding=N//2)
        self.avgpool = nn.AvgPool2d(N+1, stride=1, padding=N//2)

    def forward_one(self, repeatability) :
        return 1 - (self.maxpool(repeatability) - self.avgpool(repeatability)).mean()

    def forward(self, repeatability_map) :

        image1_repeatability, image2_repeatability = repeatability_map
        peaky_loss1 = self.forward_one(image1_repeatability)
        peaky_loss2 = self.forward_one(image2_repeatability)

        return (peaky_loss1 + peaky_loss2)/2

class APLoss(nn.Module) :
    def __init__(self, k=0.5):
        super().__init__()
        self.k = k
        self.qaploss = QAPLoss()

    def forward(self, descriptors, reliability, meta) :
        """
        Args :
            - descriptors : [image1_descriptor, image2_descriptor]
            - image1_descriptor : descriptor of image1 torch tensor shape of (batch_size, 128, H, W)
            - image2_descriptor : descriptor of image2 torch tensor shape of (batch_size, 128, H, W)
            - reliability : reliability of image1 torch tensor shape of (batch_size, 1, H, W) 
            - meta['grid'] : transformation grid (image2 to image1) 
            - meta['label'] : label for each image1 torch tensor shape of (batch_size, H, W, H, W)
        """
        image1_descriptor, image2_descriptor = descriptors
        image2_descriptor = F.grid_sample(image2_descriptor, meta['grid'])
        batch_size = image1_descriptor.shape[0]
        AP = []

        for i in range(batch_size) :
            query = image1_descriptor[i]
            db = image2_descriptor[i]
            descriptor_label = meta['label'][i]

            query = query.view(-1, query.shape[1]*query.shape[2]).transpose(1,0)
            # query shape (128, H, W) -> (H*W, 128)
            
            db = db.view(-1, db.shape[1]*db.shape[2]).transpose(1,0)
            db = db.unsqueeze(0).expand(query.shape[0], -1, -1)
            # db shape (128, H, W) -> (H*W, H*W, 128)

            descriptor_label = descriptor_label.view(descriptor_label.shape[0]*descriptor_label.shape[1], descriptor_label.shape[2]*descriptor_label.shape[3])
            # label shape (H, W, H, W) --> (H*W, H*W)

            AP.append(self.qaploss.forward(query, db, descriptor_label))

        mAP = torch.mean(torch.stack(AP))

        return mAP



            
