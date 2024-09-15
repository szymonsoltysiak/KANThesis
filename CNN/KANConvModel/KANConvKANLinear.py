from torch import nn
import torch.nn.functional as F

from KANConv import KAN_Convolutional_Layer

import sys
sys.path.append("..")
from FastKAN.KAN import KANet

class KANConvLinear(nn.Module):
    def __init__(self,device: str = 'cpu'):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = KAN_Convolutional_Layer(
            n_convs = 3,
            kernel_size = (3,3),
            padding = (1,1),
            device = device
        )
        self.bn3 = nn.BatchNorm2d(192)

        self.pool = nn.MaxPool2d(
            kernel_size=(2, 2)
        )
        
        self.flat = nn.Flatten() 

        self.fc1 = nn.Linear(3072, 512)
        self.kanet = KANet([512, 128, 43], num_knots=5, spline_order=3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.kanet(x, update_knots=True) 
        x = F.log_softmax(x, dim=1)

        return x