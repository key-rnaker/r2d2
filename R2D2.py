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
        x = image
        for layer in self.L2_layers :
            x = layer(x)

        descriptor = self.descript_layer(x)
        reliability = F.softmax(self.reliability_layer(x**2))[:,1]
        repeatability = F.softmax(self.repeatability_layer(x**2))[:,1]

        return descriptor, reliability, repeatability

    def forward(self, images) :
        descriptors = []
        reliabilities = []
        repeatabilities = []
        for img in images :
            descriptor, reliability, repeatability = self.forward_one(img)
            descriptors.append(descriptor)
            reliabilities.append(reliability)
            repeatabilities.append(repeatability)

        return descriptors, reliabilities, repeatabilities
