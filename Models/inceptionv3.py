import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, BN: bool=True, Act: nn.Module=nn.Identity()):
        super().__init__()
        self.bias = False if BN else True
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=self.bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = Act
        
    def forward(self, x):
        x = self.conv(x)
        if not self.bias:
            x = self.bn(x)
        return self.relu(x)


class ModuleB(nn.Module):
    def __init__(self, in_channels, c1x1_out, c3x3_in, c3x3_out, c5x5_in, c5x5_out, pool_proj, stride = 1):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=c1x1_out, kernel_size=1, stride=stride, padding=0, Act=nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c3x3_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c3x3_in, out_channels=c3x3_out, kernel_size=3, stride=stride, padding=1, Act=nn.ReLU(inplace=True))
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c5x5_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_in, out_channels=c5x5_out, kernel_size=3, stride=1, padding=1, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=3, stride=stride, padding=1, Act=nn.ReLU(inplace=True))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=stride, padding=0, Act=nn.ReLU(inplace=True))
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3x3_in, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c3x3_in, out_channels=c3x3_out*4, kernel_size=3, stride=stride, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = x * x5
        return x
    

class ModuleA(nn.Module):
    def __init__(self, in_channels, c1x1_out, c3x3_in, c3x3_out, c5x5_in, c5x5_out, pool_proj, kernel_size:int =7):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=c1x1_out, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c3x3_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c3x3_in, out_channels=c3x3_out, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2), Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c3x3_out, out_channels=c3x3_out, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size//2, 0), Act=nn.ReLU(inplace=True))
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c5x5_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_in, out_channels=c5x5_out, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2), Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size//2, 0), Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size//2), Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=(kernel_size, 1), stride=1, padding=(kernel_size//2, 0), Act=nn.ReLU(inplace=True))
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3x3_in, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c3x3_in, out_channels=c3x3_out*4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = x*x5
        return x
    
    
class ModuleC(nn.Module):
    def __init__(self, in_channels, c1x1_out, c3x3_in, c3x3_out, c5x5_in, c5x5_out, pool_proj):
        super().__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=c1x1_out, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        self.branch2 = ConvBlock(in_channels=in_channels, out_channels=c3x3_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        self.branch2_1 = ConvBlock(in_channels=c3x3_in, out_channels=c3x3_out//2, kernel_size=(1, 3), stride=1, padding=(0, 1), Act=nn.ReLU(inplace=True))
        self.branch2_2 = ConvBlock(in_channels=c3x3_in, out_channels=c3x3_out//2, kernel_size=(3, 1), stride=1, padding=(1, 0), Act=nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c5x5_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_in, out_channels=c5x5_out//2, kernel_size=3, stride=1, padding=1, Act=nn.ReLU(inplace=True)),
        )
        self.branch3_1 = ConvBlock(in_channels=c5x5_out//2, out_channels=c5x5_out//2, kernel_size=(1, 3), stride=1, padding=(0, 1), Act=nn.ReLU(inplace=True))
        self.branch3_2 = ConvBlock(in_channels=c5x5_out//2, out_channels=c5x5_out//2, kernel_size=(3, 1), stride=1, padding=(1, 0), Act=nn.ReLU(inplace=True))
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3x3_in, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=c3x3_in, out_channels=c3x3_out*4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x21 = self.branch2_1(x2)
        x22 = self.branch2_2(x2)
        x3 = self.branch3(x)
        x31 = self.branch3_1(x3)
        x32 = self.branch3_2(x3)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x = torch.cat([x1, x21, x22, x31, x32, x4], dim=1)

        x = x*x5
        return x
    

class ModuleD(nn.Module):
    def __init__(self, in_channels, c3x3_in, c3x3_out, c5x5_in, c5x5_out):
        super().__init__()
        self.branch1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c3x3_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c3x3_in, out_channels=c3x3_out, kernel_size=3, stride=2, padding=0, Act=nn.ReLU(inplace=True))
        )
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=c5x5_in, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_in, out_channels=c5x5_out, kernel_size=3, stride=1, padding=1, Act=nn.ReLU(inplace=True)),
            ConvBlock(in_channels=c5x5_out, out_channels=c5x5_out, kernel_size=3, stride=2, padding=0, Act=nn.ReLU(inplace=True)),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    

class AuxClassifier(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, BN: bool=True):
        super().__init__()
        self.bias = False if BN else True
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = ConvBlock(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0, BN=BN, Act=nn.ReLU(inplace=True))
        self.conv5x5 = ConvBlock(in_channels=128, out_channels=out_channels, kernel_size=5, stride=1, padding=0, BN=BN)
        self.globalpool = nn.AdaptiveAvgPool2d(1) # 防止输入尺寸不是299的情况
        self.fc = nn.Linear(in_features=1024, out_features=n_classes, bias=True)

    def forward(self, x):
        return self.fc(torch.flatten(self.globalpool(self.conv5x5(self.conv1x1(self.avgpool(x)))), start_dim=1))
    
    
class Inception(nn.Module):
    def __init__(self, version = 3, in_channels = 1, num_classes = 3):
        super().__init__()
        assert version in (2, 3)
        self.bn = True if version == 3 else False
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0, Act=nn.ReLU(inplace=True))
        self.conv2 = ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        self.conv3 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, Act=nn.ReLU(inplace=True))

        self.conv4 = ConvBlock(in_channels=64, out_channels=80, kernel_size=1, stride=1, padding=0, Act=nn.ReLU(inplace=True))
        self.conv5 = ConvBlock(in_channels=80, out_channels=192, kernel_size=3, stride=1, padding=0, Act=nn.ReLU(inplace=True))

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.X3inception = nn.Sequential(
            ModuleA(in_channels=192, c1x1_out=48, c3x3_in=108, c3x3_out=48, c5x5_in=108, c5x5_out=48, pool_proj=48),
            ModuleA(in_channels=192, c1x1_out=48, c3x3_in=108, c3x3_out=48, c5x5_in=108, c5x5_out=48, pool_proj=48),
            # ModuleA(in_channels=288, c1x1_out=72, c3x3_in=144, c3x3_out=72, c5x5_in=144, c5x5_out=72, pool_proj=72),
            ModuleD(in_channels=192, c3x3_in=144, c3x3_out=48, c5x5_in=144, c5x5_out=48),
        )
        # self.auxclassifier = AuxClassifier(in_channels=768, out_channels=1024, n_classes=n_classes, BN=self.bn)
        self.X5inception = nn.Sequential(
            ModuleB(in_channels=288, c1x1_out=60, c3x3_in=144, c3x3_out=60, c5x5_in=144, c5x5_out=60, pool_proj=60),
            ModuleB(in_channels=240, c1x1_out=60, c3x3_in=120, c3x3_out=60, c5x5_in=120, c5x5_out=60, pool_proj=60),
            ModuleB(in_channels=240, c1x1_out=60, c3x3_in=120, c3x3_out=60, c5x5_in=120, c5x5_out=60, pool_proj=60),
            ModuleB(in_channels=240, c1x1_out=60, c3x3_in=120, c3x3_out=60, c5x5_in=120, c5x5_out=60, pool_proj=60),
            # ModuleB(in_channels=240, c1x1_out=192, c3x3_in=120, c3x3_out=192, c5x5_in=120, c5x5_out=192, pool_proj=192, stride=2),
            ModuleD(in_channels=240, c3x3_in=120, c3x3_out=264, c5x5_in=120, c5x5_out=264),
        )
        self.X2inception = nn.Sequential(
            ModuleC(in_channels=768, c1x1_out=256, c3x3_in=480, c3x3_out=256, c5x5_in=480, c5x5_out=256, pool_proj=256),
            ModuleC(in_channels=1024, c1x1_out=512, c3x3_in=1024, c3x3_out=512, c5x5_in=1024, c5x5_out=512, pool_proj=512),
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=2048, out_features=num_classes, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.conv1(x)
        # print('# Conv1 output shape:', x.shape)
        x = self.conv2(x)
        # print('# Conv2 output shape:', x.shape)
        x = self.conv3(x)
        # print('# Conv3 output shape:', x.shape)
        x = self.conv4(x)
        # print('# Conv4 output shape:', x.shape)
        x = self.conv5(x)
        # print('# Conv5 output shape:', x.shape)
        x = self.pool1(x)
        # print('# Pool1 output shape:', x.shape)
        x = self.X3inception(x)
        # print('# X3inception output shape:', x.shape)
        # aux = self.auxclassifier(x)
        # print('# AuxClassifier output shape:', aux.shape)
        x = self.X5inception(x)
        # print('# X5inception output shape:', x.shape)
        x = self.X2inception(x)
        # print('# X2inception output shape:', x.shape)
        x = self.globalpool(x)
        # print('# Globalpool output shape:', x.shape)
        x = self.fc(x)
        # print('# FC output shape:', x.shape)
        x = self.softmax(x)
        # print('# Softmax output shape:', x.shape)
        return x
    
    
# inputs = torch.randn(4, 1, 224, 224)

# cnn = Inception(version=3, in_channels = 1, num_classes = 3)
# # print(cnn)
# outputs = cnn(inputs)