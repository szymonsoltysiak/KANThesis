import sys
sys.path.append("..")
from FastKAN.KAN import KANet

import torch.nn as nn
import torch.nn.functional as F

class CNNKan(nn.Module):
    def __init__(self):
        super(CNNKan, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.kanet = KANet([128 * 4 * 4, 512, 128, 43], num_knots=5, spline_order=3)

    def forward(self, x, update_knots=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4) 

        x = self.kanet(x, update_knots=update_knots)
        
        return x