import torch
from torch import nn, einsum
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        return out

class DscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DscBlock, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0, groups=in_channels) # depthwise conv
        self.norm1 = nn.BatchNorm2d(in_channels)

        self.pwconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.norm1(self.dwconv(x))
        out = F.relu(out)

        out = self.norm2(self.pwconv(x))
        out = F.relu(out)

        return out

class DSCNN(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3,padding=1)
        self.dwconv1 = DscBlock(in_channels=32,out_channels=32) 

        self.conv2 = ConvBlock(in_channels=32,out_channels=64,kernel_size=1,padding=0)
        self.dwconv2 = DscBlock(in_channels=64,out_channels=64)

        self.conv3 = ConvBlock(in_channels=64,out_channels=128,kernel_size=1,padding=0)

        self.global_average_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(6272, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.dwconv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.dwconv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.global_average_pool(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

inputs = torch.randn(4, 1, 224, 224)

cnn = DSCNN(in_channels = 1, num_classes = 3)
print(cnn)
outputs = cnn(inputs)


