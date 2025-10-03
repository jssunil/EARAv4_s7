import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from layers import SeparableConv2d

  

class CIFAR10_C1C2C3C40(nn.Module):
    def __init__(self, num_classes=10, dropout_value=0.05):
        super().__init__()

        # C1: 32ch, downsample with 3x3 s2 (no MaxPool)
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),

            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),  # downsample-1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
        )

        # C2: 48ch, includes Dilated conv, then stride-2 downsample
        self.c2 = nn.Sequential(
            nn.Conv2d(32, 48, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value),

            nn.Conv2d(48, 48, 3, stride=1, padding=2, dilation=2, bias=False),  # Dilated
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value),

            nn.Conv2d(48, 48, 3, stride=2, padding=1, bias=False),  # downsample-2
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value),
        )

        # C3: uses Depthwise Separable conv, then stride-2 downsample
        self.c3 = nn.Sequential(
            SeparableConv2d(48, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),  # Depthwise Separable
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),  # downsample-3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        # C4-0 head: light projection + conv, then GAP and classifier
        self.c40 = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),

            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.c1(x)
        z = self.convblock1(x)
        z = self.convblock2(z)
        z = self.transitionblock1(z)
        z = F.relu(z)
        z = self.convblock3(z)

        y = self.convblock4(z)
        y = self.convblock5(y)
        y = self.transitionblock2(y)
        y = F.relu(y)
        y = self.convblock6(y)

        y = self.convblock7(y)
        y = self.output(y)
        y = y.view(-1, 10)

        return F.log_softmax(y, dim=-1)
