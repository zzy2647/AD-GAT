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


class AttentionBlock(nn.Module):
    def __init__(self, in_features, out_features, heads, patch, channels, concat=True):
        super(AttentionBlock, self).__init__()
        self.heads = heads
        self.concat = concat

        self.to_W = nn.Linear(in_features, heads * out_features, bias=False)
        self.to_a_self = [nn.Linear(out_features, 1, bias=False) for _ in range(heads)]
        self.to_a_neighbor = [nn.Linear(out_features, 1, bias=False) for _ in range(heads)]
        self.to_patch = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch, p2=patch),
            nn.Linear(patch*patch*channels, in_features),
        )

        self.to_out = nn.Linear(out_features*heads, in_features) if concat else None
        self.layer_norm = nn.LayerNorm(in_features)

    def forward(self, token, feature):
        # token (b d) feature (b d w h)
        device = token.device

        feature_h = self.to_patch(feature)
        feature_h = self.to_W(feature_h)
        token_h = self.to_W(token)

        token_h = rearrange(token_h, 'b (h d) -> b h d', h=self.heads)
        feature_h = rearrange(feature_h, 'b n (h d) -> b h n d', h=self.heads)

        out = torch.zeros_like(token_h).to(device)
        for i in range(self.heads):
            to_a_self = self.to_a_self[i].to(device)
            to_a_neighs = self.to_a_neighbor[i].to(device)
            attn_for_self = to_a_self(token_h[:, i])
            attn_for_neighs = to_a_neighs(feature_h[:, i])
            attn = attn_for_self.unsqueeze(1) + attn_for_neighs

            attn = F.leaky_relu(attn, negative_slope=0.2)
            attn = F.softmax(attn, dim=-1)

            out[:, i] = einsum('b n , b n d -> b d', attn.squeeze(), feature_h[:, i])

        if self.concat:
            out = 0.5 * self.to_out(repeat(out, 'b h d -> b (h d)')) + 0.5 * token
        else:
            out = 0.5 * torch.mean(out, dim=1) + 0.5 * token

        out = self.layer_norm(0.5*out + 0.5*token)

        return out


class ConvGraphLayer(nn.Module):
    def __init__(self, in_channesl, out_channels, stride, patch, dims, heads):
        super(ConvGraphLayer, self).__init__()
        self.conv = ConvBlock(in_channesl, out_channels, stride=stride)
        self.graph = AttentionBlock(dims, dims, heads, patch, out_channels)

    def forward(self, token, feature):
        feature = self.conv(feature)
        token = self.graph(token, feature)
        return token, feature


class ConvGAT(nn.Module):
    def __init__(self, token_dims, num_classes, heads=2):
        super(ConvGAT, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, token_dims))

        self.layer = nn.ModuleList([
            ConvGraphLayer(1, 16, stride=1, patch=8, dims=token_dims, heads=heads),
            ConvGraphLayer(16, 32, stride=2, patch=4, dims=token_dims, heads=heads),
            ConvGraphLayer(32, 64, stride=2, patch=2, dims=token_dims, heads=heads),
            ConvGraphLayer(64, 128, stride=2, patch=1, dims=token_dims, heads=heads),
        ])

        self.classifier = nn.Linear(token_dims, num_classes)

    def forward(self, x):
        b, c, w, h = x.shape
        token = repeat(self.cls_token, '() d -> b d', b=b)
        for l in self.layer:
            token, x = l(token, x)
        # token, feature = self.layer(token, x)
        return self.classifier(token)


if __name__ == '__main__':
    model = ConvGAT(256, 2)
    feature = torch.randn((30, 1, 184, 224))
    y = model(feature)
    print('done')