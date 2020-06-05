import torch.nn as nn

class L2(nn.Module) :
    def __init__(self) :
        super().__init__()
        self.L2_layers = nn.ModuleList([])
        self.add_conv(3, 32, 3, 1)
        self.add_conv(32, 32, 3, 1)
        self.add_conv(32, 64, 3, 2)
        self.add_conv(64, 64, 3, 2)
        self.add_conv(64, 128, 3, 4)
        self.add_conv(128, 128, 3, 4)
        self.add_conv(128, 128, 2, 4)
        self.add_conv(128, 128, 2, 4)
        self.add_conv(128, 128, 2, 4)

    def add_conv(self, inch, outch, kernel, dilation) :
        stride = 1
        self.L2_layers.append(nn.Conv2d(inch, outch, kernel, stride, padding=dilation*(kernel-1)//2, dilation=dilation))
        self.L2_layers.append(nn.BatchNorm2d(outch))
        self.L2_layers.append(nn.ReLU(True))
