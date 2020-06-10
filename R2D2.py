import torch
import torch.nn as nn
import torch.nn.functional as F
from L2 import L2

class R2D2(L2) :
    def __init__(self) :
        super().__init__()
        self.descript_layer = F.normalize(self.out_dim)
        self.reliability_layer = nn.Conv2d(self.out_dim, 2, 1)
        self.repeatability_layer = nn.Conv2d(self.out_dim, 2, 1)

    def forward_one(self, image) :
        """
        Args :
            - image : input image torch tensor shape (3, H, W)

        Return :
            - descriptor : descriptor torch tensor shape (128, H, W)
            - reliability : reliability heatmap torch tensor shape (1, H, W)
            - repeatability : repeatability heatmap torch tensor shape (1, H, W)
        """
        x = image
        for layer in self.L2_layers :
            # x shape (128, H, W)
            x = layer(x)

        # descriptor shape (128, H, W)
        descriptor = self.descript_layer(x)
        # reliability shape (1, H, W)
        reliability = F.softmax(self.reliability_layer(x**2))[:,1]
        # repeatability shape (1, H, W)
        repeatability = F.softmax(self.repeatability_layer(x**2))[:,1]

        return descriptor, reliability, repeatability

    def forward(self, images) :
        """
        Args :
            - images : input images torch tensor shape (batch_size, 3, H, W)

        Return :
            - descriptors : image descriptors torch tensor shape (batch_size, 128, H, W)
            - reliabilities : reliability heatmaps torch tensor shape (batch_size, 1, H, W)
            - repeatabilities : repeatability heatmaps torch tensor shape (batch_size, 1, H, W)
        """
        descriptors = []
        reliabilities = []
        repeatabilities = []
        for img in images :
            descriptor, reliability, repeatability = self.forward_one(img)
            descriptors.append(descriptor)
            reliabilities.append(reliability)
            repeatabilities.append(repeatability)

        return torch.stack(descriptors), torch.stack(reliabilities), torch.stack(repeatabilities)
