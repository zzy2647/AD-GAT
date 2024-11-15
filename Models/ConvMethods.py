# import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from vit_pytorch import ViT

import math
from functools import partial
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange

# from Models.utils.tools import merge_pre_bn
# from utils.tools import merge_pre_bn

class Block(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, is_shortcut=False):
        super(Block,self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.is_shortcut = is_shortcut
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, groups=32,
                                   bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if is_shortcut:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_shortcut:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x
 
class Resnext(nn.Module):
    def __init__(self,num_classes,layer=[3,4,6,3]):
        super(Resnext,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(64,256,1,num=layer[0])
        self.conv3 = self._make_layer(256,512,2,num=layer[1])
        self.conv4 = self._make_layer(512,1024,2,num=layer[2])
        self.conv5 = self._make_layer(1024,2048,2,num=layer[3])
        self.global_average_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048,num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_average_pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    def _make_layer(self,in_channels,out_channels,stride,num):
        layers = []
        block_1=Block(in_channels, out_channels,stride=stride,is_shortcut=True)
        layers.append(block_1)
        for i in range(1, num):
            layers.append(Block(out_channels,out_channels,stride=1,is_shortcut=False))
        return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self,num_classes=3,**kwargs):
        super(VGG16, self).__init__()
        vgg16=models.vgg16_bn(pretrained=False)
        
        vgg16.features[0]=nn.Conv2d(1 ,64 ,3 ,1 ,1)
        num_fc = vgg16.classifier[6].in_features
        vgg16.classifier[6]=nn.Linear(num_fc,num_classes)
        self.vgg16=vgg16

    def forward(self,x):
        return self.vgg16(x)

class GoogLeNet(nn.Module):
    def __init__(self,num_classes=3,**kwargs):
        super(GoogLeNet, self).__init__()
        googlenet=models.googlenet(pretrained=False)

        googlenet.conv1.conv=nn.Conv2d(1 ,64 ,7 ,2 ,3)
        # num_fc = googlenet.classifier[6].in_features
        googlenet.fc=nn.Linear(1024,num_classes)
        self.googlenet=googlenet

    def forward(self,x):
        return self.googlenet(x)
class ResNet(nn.Module):
    def __init__(self,num_classes=3,**kwargs):
        super(ResNet, self).__init__()
        resnet=models.resnet50(pretrained=False)

        resnet.conv1=nn.Conv2d(1 ,64 ,7 ,2 ,3)
        # resnet.layer1[0].conv1=nn.Conv2d(64 ,64 ,11 ,1 ,5,bias=False)
        resnet.fc=nn.Linear(2048,num_classes)
        self.resnet=resnet

    def forward(self,x):
        return self.resnet(x)

class MobileNetV2(nn.Module):
    def __init__(self,num_classes=3,**kwargs):
        super(MobileNetV2, self).__init__()
        mobilenet=models.mobilenet_v2(pretrained=False)

        mobilenet.features[0][0]=nn.Conv2d(1 ,32 ,3 ,2 ,1)
        mobilenet.classifier[1]=nn.Linear(1280,num_classes)
        self.mobilenet=mobilenet

    def forward(self,x):
        return self.mobilenet(x)       
class VIT_b_16(nn.Module):
    def __init__(self,num_classes=3,**kwargs):
        super(VIT_b_16, self).__init__()
        Num_classes = num_classes
        vit_b_16=ViT(
            image_size = 224,
            patch_size = 16,
            num_classes = Num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            channels = 1
            )

        self.vit_b_16=vit_b_16

    def forward(self,x):
        return self.vit_b_16(x)    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size = 3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        return out 
class AlzNet(nn.Module):
     def __init__(self, in_channels = 1, num_class = 3):
        super(AlzNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64, stride=1, kernel_size= 9)
        self.conv2 = ConvBlock(64, 64, stride=1, kernel_size= 7)
        self.conv3 = ConvBlock(64, 64, stride=1, kernel_size= 5)
        self.conv4 = ConvBlock(64, 32, stride=1, kernel_size= 5)
        self.conv5 = ConvBlock(32, 32, stride=1, kernel_size= 3)

        self.dense1 = nn.Linear(800,121)
        self.drop = nn.Dropout()
        self.dense2 = nn.Linear(121,num_class)
     def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = torch.flatten(x,1)
        x = F.relu(self.dense1(x))
        x = self.drop(x)
        x = self.dense2(x)

        return x

class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1,
            NORM_EPS = 1e-5):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
NORM_EPS = 1e-5
class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out
'''
class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
        
'''
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        u = x.clone()        
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn #u * 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Basicblock(nn.Module):
    def __init__(self, dim, mlp_ratio = 2, drop_path = 0, NORM_EPS = 1e-5):
        super().__init__()
        self.dwconv = LKA(dim)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_path)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim, eps=NORM_EPS)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        x1 = self.dwconv(x)
        x = x1.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        # x = input + self.drop_path(x) * self.lk(x1)
        x = input + self.drop_path(x) * self.mlp(x1)
        
        return x
class H_ConvNet(nn.Module):
    def __init__(self,in_chans=1, num_classes=3, depths=[2, 3, 4, 2], dims=[32, 64, 192, 384], drop_path_rate=0.):
        super(H_ConvNet, self).__init__()
        self.depth = depths
        self.downsample_layers = nn.ModuleList()
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        # )
        stem = nn.Sequential(
            nn.Conv2d(in_chans,dims[0],kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample_layers.append(stem)
        
        for i in range(len(depths)-1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Basicblock(dim=dims[i], drop_path=dp_rates[cur + j]) 
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.classifier = nn.Linear(dims[-1], num_classes)
    def forward_features(self, x):
        for i in range(len(self.depth)):
            x = self.downsample_layers[i](x)
            # print(x.shape)
            x = self.stages[i](x)
        # print(x.shape)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)
        x = self.classifier(x)
        
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


if __name__ == '__main__':
    x = torch.randn((4, 1, 224, 224))
    model = AlzNet()
    print(model)
    y = model(x)
    print(y.shape)

