import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.leaky_relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, channels):
        super(ConvNet, self).__init__()
        # channels = [16, 32, 64, 128]
        self.conv_in = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels[0])

        self.layer1 = nn.Sequential(
            ConvBlock(channels[0], channels[0], 1),
            ConvBlock(channels[0], channels[0], 1),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(channels[0], channels[1], 2),
            ConvBlock(channels[1], channels[1], 1),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(channels[1], channels[2], 2),
            ConvBlock(channels[2], channels[2], 1),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(channels[2], channels[3], 2),
            ConvBlock(channels[3], channels[3], 1),
        )

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        return out1, out2, out3, out4


class VariableNodesAttentionLayer(nn.Module):
    def __init__(self, nodes_num, f_dims, a_dims, v_dims, heads=1):
        super(VariableNodesAttentionLayer, self).__init__()
        self.heads = heads
        self.nodes_num = nodes_num
        self.v_dims = v_dims
        self.scale = a_dims ** -0.5

        self.nodes = nn.Parameter(torch.randn(nodes_num, a_dims))

        self.to_a = nn.Linear(f_dims, a_dims * heads, bias=False)
        self.to_v = nn.Linear(f_dims, v_dims * heads)
        self.layer_norm = nn.LayerNorm(v_dims * heads)

    def forward(self, x):
        b, n, d = x.shape

        # nodes = repeat(self.nodes, '() n d -> b n d', b=b)

        a = self.to_a(x)
        v = self.to_v(x)
        a = rearrange(a, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        out = torch.empty((b, self.heads, self.nodes_num, self.v_dims)).to(x.device)

        for i in range(self.heads):
            attn = torch.einsum('i d, b j d -> b i j', self.nodes, a[:, i]) * self.scale
            attn = F.softmax(attn, dim=-1)
            out[:, i] = torch.einsum('b i j, b j d -> b i d', attn, v[:, i])

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.layer_norm(out)

        return out


class VariableNodesGraph(nn.Module):
    def __init__(self, nodes_list, f_dims, heads=2):
        super(VariableNodesGraph, self).__init__()
        self.layer_num = len(nodes_list)
        self.layer = nn.ModuleList()
        for num in nodes_list:
            self.layer.append(nn.Dropout(0.5))
            self.layer.append(VariableNodesAttentionLayer(num, f_dims, 64, f_dims // heads, heads))

    def forward(self, x):
        for f in self.layer:
            x = f(x)
        return x


class VariableGAT(nn.Module):
    def __init__(self, num_classes, heads=2):
        super(VariableGAT, self).__init__()
        self.conv_net = ConvNet(channels=[16, 32, 64, 128])
        self.gat = VariableNodesGraph([161, 40, 1], 128, heads)
        self.classifier = nn.Linear(128, num_classes)

        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1)
        )

    def forward(self, x):

        out1, out2, out3, out4 = self.conv_net(x)
        out = self.to_patch(out4)
        feature = self.gat(out)
        feature = torch.squeeze(feature, 1)
        return self.classifier(feature)


if __name__ == '__main__':
    x = torch.randn((30, 1, 184, 224))
    model = VariableGAT(3)
    y = model(x)
    print('done')