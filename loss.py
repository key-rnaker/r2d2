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
            - heatmap : repeatability heatmap tensor shape of (1, 1, H, W)

        Return :
            - patches : flatten patches tensor shape of (1, (H-(N-1) * W-(N-1)) , N*N)
        """
        patches = nn.Unfold(self.N)
        patches = patches(heatmap).transpose(1,2)
        patches = F.normalize(patches, p=2, dim=2)
        return patches

    def forward_one(self, rep_map1, rep_map2, meta) :
        """
        Args :
            - rep_map1 : repeatability heatmap tensor shape of (1, H, W)
            - rep_map2 : repeatability heatmap tensor shape of (1, H2, W2)
            - meata['grid'] = grid rep_map2 --> rep_map1 tensor shape of (1, H, W, 2)
        """

        #transform_rep_map = F.grid_sample(rep_map2.unsqueeze(0), meta['grid'], mode='bilinear', padding_mode='border')
        transform_rep_map = F.grid_sample(rep_map2.unsqueeze(0), meta['grid'])
        if 'mask' in meta :
            rep_map1 = rep_map1 * meta['mask']
        image1_patches = self.extract_pacthes(rep_map1.unsqueeze(0))
        image2_patches = self.extract_pacthes(transform_rep_map)
        cosim = (image1_patches * image2_patches).sum(dim=2)

        return 1 *( cosim[cosim!=0].shape[0] / cosim.shape[1]) - cosim.mean()


    def forward(self, repeatability_map, metas) :
        """
        Args :
            - repeatability_map : [image1_repeatability, image2_repeatability]
            - image1_repeatability : repeatability heatmap tensor shape of (batch_size, 1, H, W)
            - image2_repeatability : repeatability heatmap tensor shape of (batch_size, 1, H, W)
            - metas : metas
        """
        #grid shape = [batch_size, image1_height, image1_width, 2]

        image1_repeatability, image2_repeatability = repeatability_map
        B = image1_repeatability.shape[0]

        loss = []
        for b in range(B) :
            loss.append(self.forward_one(image1_repeatability[b], image2_repeatability[b], metas[b]))

        return torch.mean(torch.stack(loss))

class PeakyLoss(nn.Module) :
    def __init__(self, patch_size=16):
        super().__init__()
        self.N = patch_size
        self.maxpool = nn.MaxPool2d(self.N+1, stride=1, padding=self.N//2)
        self.avgpool = nn.AvgPool2d(self.N+1, stride=1, padding=self.N//2)

    def forward_one(self, repeatability) :
        return 1 - (self.maxpool(repeatability.unsqueeze(0)) - self.avgpool(repeatability.unsqueeze(0))).mean()

    def forward(self, repeatability_map) :
        """
        Args :
            - repeatability_map : [image1_repeatability, image2_repeatability]
            - image1_repeatability : repeatability heatmap torch tensor shape of (batch_size, 1, H, W)
            - image2_repeatability : repeatability heatmap torch tensor shape of (batch_size, 1, H, W)
        """
        image1_repeatability, image2_repeatability = repeatability_map
        B = image1_repeatability.shape[0]
        
        peaky_loss1 = []
        peaky_loss2 = []
        for b in B :
            peaky_loss1.append(self.forward_one(image1_repeatability[b]))
            peaky_loss2.append(self.forward_one(image2_repeatability[b]))

        peaky_loss1 = torch.mean(torch.stack(peaky_loss1))
        peaky_loss2 = torch.mean(torch.stack(peaky_loss2))

        return (peaky_loss1 + peaky_loss2)/2

class APLoss(nn.Module) :
    def __init__(self, k=0.5):
        super().__init__()
        self.k = k
        self.qaploss = QAPLoss()

    def forward_one(self, descriptor1, descriptor2, reliability, meta) :
        """
        Args :
            - descriptor1 : descriptor of image1 torch tensor shape of (128, H, W)
            - descriptor2 : descriptor of image2 torch tensor shape of (128, H2, W2)
            - reliability : reliability heatmap of image1 torch tensor shape of (1, H, W)
        """

        H, W = descriptor1.shape[1:]
        query = descriptor1.view(-1, H*W).transpose(1,0)
        # query shape (128, H, W) -> (H*W, 128)

        db = F.grid_sample(descriptor2.unsqueeze(0), meta['grid']).squeeze(0)
        db = db.view(-1, H*W).transpose(1,0)
        # db shape (H*W, 128)

        reliability = reliability.view(-1)
        # reliability shape (H*W)

        flatten_mask = (meta['mask'][0] == 1).view(-1)
        # flatten_mask shape (H*W)

        APQ = []
        for i in range(H*W) :
            y = i % W
            x = i // W
            descriptor_label = torch.zeros([H,W])
            descriptor_label[max(0, x-4) : min(H, x+5), max(0, y-4) : min(W, y+5)] = 1
            db = db[flatten_mask]
            descriptor_label = descriptor_label.view(-1)[flatten_mask]
            ap = self.qaploss.forward_one(query[i], db, descriptor_label)
            APQ.append(1 - (ap * reliability[i] + self.k*(1 - reliability[i])))
            del x, y, descriptor_label

        return  torch.mean(torch.stack(APQ))

    def forward(self, descriptors, reliability, metas) :
        """
        Args :
            - descriptors : [image1_descriptor, image2_descriptor]
            - image1_descriptor : descriptor of image1 torch tensor shape of (batch_size, 128, H, W)
            - image2_descriptor : descriptor of image2 torch tensor shape of (batch_size, 128, H, W)
            - reliability : reliability of image1 torch tensor shape of (batch_size, 1, H, W) 
            - metas : meta (batch_size, meta)
        """
        image1_descriptor, image2_descriptor = descriptors
        #image2_descriptor = F.grid_sample(image2_descriptor, meta['grid'])
        batch_size = image1_descriptor.shape[0]
        AP = []

        for b in range(batch_size) :
            AP.append(self.forward_one(image1_descriptor[b], image2_descriptor[b], reliability[b], metas[b]))

        return torch.mean(torch.stack(AP))



            
