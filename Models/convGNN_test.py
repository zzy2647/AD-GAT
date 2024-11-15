import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class FeatureMapConcat(nn.Module):
    def __init__(self, channels):
        super(FeatureMapConcat, self).__init__()
        self.to_patch1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=8, p2=8),
            nn.Linear(8*8*channels[0], channels[3]),
        )
        self.to_patch2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),
            nn.Linear(4*4*channels[1], channels[3]),
        )
        self.to_patch3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),
            nn.Linear(2*2*channels[2], channels[3]),
        )
        self.to_patch4 = nn.Sequential(
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        temp1 = self.to_patch1(x[0])
        temp2 = self.to_patch2(x[1])
        temp3 = self.to_patch3(x[2])
        temp4 = self.to_patch4(x[3])
        out_concat = torch.cat([temp4, temp3, temp2, temp1], dim=2)

        return out_concat


class ConvNet(nn.Module):
    def __init__(self, in_channels):
        super(ConvNet, self).__init__()
        channels = [16, 32, 64, 128]
        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
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

        self.feature_concat = FeatureMapConcat(channels)

    def forward(self, x):
        out = F.relu(self.bn_in(self.conv_in(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out_concat = self.feature_concat((out1, out2, out3, out4))

        return out_concat, out4
''' 
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x       
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

class GroupAttention(nn.Module):
    """
    LSA: self attention within a group
    """
    def __init__(self, dim, head_dim=32, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        # assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.qkv(x).reshape(B, total_groups, -1, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # B, hw, ws*ws, 3, n_head, head_dim -> 3, B, hw, n_head, ws*ws, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(
            attn)  # attn @ v-> B, hw, n_head, ws*ws, head_dim -> (t(2,3)) B, hw, ws*ws, n_head,  head_dim
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MHAttention(nn.Module):
    def __init__(self, dim, head_dim=32, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = self.dim // head_dim

        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, int(C // self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GMlp(nn.Module):
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
        self.mlp = GMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_path)
        # self.mlp = SMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop_path)
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

class Basicblock2(nn.Module):
    def __init__(self, dim, mlp_ratio = 2, drop_path = 0,drop_ratio = 0.75, head_dim = 32, NORM_EPS = 1e-5, sr_ratio = 1, ws = 7):
        super().__init__()
        # self.dwconv = LKA(dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # self.mhattn = MHAttention(dim, head_dim=head_dim, attn_drop= drop_path, proj_drop= drop_path)
        self.mhattn = GroupAttention(dim, head_dim = 32, attn_drop= drop_path, proj_drop= drop_path, ws=ws)

        self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim, eps=NORM_EPS)

        self.mhsa_drop_path = DropPath(drop_path * drop_ratio) if drop_path > 0. else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        _, _, H, _ = x.shape
        # print(x.shape)
        x0 = self.dwconv(x)
        x1 = x0.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x1)
        _, H1, W1, _ = x.shape
        # print(x.shape)
        x = rearrange(x, "b h w c -> b (h w) c")
        # x = self.mhsa_drop_path(self.mhattn(x))
        x = self.mhsa_drop_path(self.mhattn(x,H1,W1))  # group attention
        x2 = x1 + rearrange(x, "b (h w) c -> b h w c", h = H)
        x = self.pwconv1(x2)
        
        x = self.act(x)
        x = self.pwconv2(x)
        x3 = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x3 = self.norm2(x3)
        # print(input.shape,self.drop_path(x).shape)
        out = input + self.drop_path(x3) 
        return out
class H_ConvNet(nn.Module):
    def __init__(self,in_chans=1, num_classes=3, mlp_ratios=[2, 2, 2, 2], depths=[3, 3, 3, 3], dims=[64, 128, 256, 512], drop_path_rate=0.):
        super(H_ConvNet, self).__init__()
        self.depth = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans,dims[0],kernel_size=13,stride=2,padding=5,bias=False),
            nn.BatchNorm2d(dims[0]),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample_layers.append(stem)
        
        for i in range(len(depths)-1):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    # nn.BatchNorm2d(dims[i], eps=1e-5),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
     
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j])
                # *[Basicblock(dim=dims[i], mlp_ratio=mlp_ratios[i], drop_path=dp_rates[cur + j]) 
                for j in range(depths[i])],
                # Basicblock2(dim = dims[i], drop_path=dp_rates[i], ws = 7) if (i == 1 or i == 3) else nn.Identity(),
                # Basicblock2(dim = dims[i], drop_path=dp_rates[i]) if (i == 1 or i == 3) else nn.Identity(),
                # Basicblock2(dim = dims[i], drop_path=dp_rates[i])
                
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.classifier = nn.Linear(dims[-1], num_classes)
        # self.to_patch = nn.Sequential(
        #     Rearrange('b c h w -> b (h w) c'))
    def forward_features(self, x):
        for i in range(len(self.depth)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # print(x.shape)
        # return x
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)
        x = self.classifier(x)
        # x = self.to_patch(x)
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

class AttentionLayer(nn.Module):
    def __init__(self, dim, heads=2):
        super(AttentionLayer, self).__init__()

        self.heads = heads
        self.scale = 64 ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, 64 * heads, bias=False)
        self.to_k = nn.Linear(dim, 64 * heads, bias=False)
        self.to_v = nn.Linear(dim, dim * heads)
        self.to_out = nn.Linear(dim*heads, dim)

    def forward(self, x):
        # b, n, _, h = *x.shape, self.heads
        q = self.to_q(x[:, 0])
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b (h d) -> b h d', h=self.heads)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (k, v))
        dots = einsum('b h d, b h i d -> b h i', q, k) * self.scale

        _mean = torch.mean(dots, dim=-1, keepdim=True)
        mask = (dots >= _mean)
        mask[:, :, 0] = True
        dots[~mask] = torch.tensor(float("-inf")).to(x.device)

        _max = torch.max(dots, dim=-1, keepdim=True)[0]
        dots = dots - _max
        attn = F.softmax(dots, dim=-1)  # self.attend(dots)

        token = einsum('b h i, b h i d -> b h d', attn, v)
        token = rearrange(token, 'b h d -> b (h d)').unsqueeze(1)
        v = rearrange(v, 'b h n d -> b n (h d)')

        out = torch.cat([token, v[:, 1:]], dim=1)
        return self.to_out(out)


class GraphNet(nn.Module):
    def __init__(self, dim, num_node, layer_depth=2):
        super(GraphNet, self).__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_node + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.layer = nn.ModuleList([nn.Sequential(
            nn.Dropout(0.5),
            AttentionLayer(dim),
            # MHAttention(dim=dim),
            nn.LayerNorm(dim),
        ) for _ in range(layer_depth)])

    def forward(self, x):
        b, n, _ = x.shape
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x += self.pos_embedding[:, :(n+1)]
        for l in self.layer:
            x = l(x)

        return x


class ConvGNN(nn.Module):
    def __init__(self, num_classes):
        super(ConvGNN, self).__init__()
        self.convNet = H_ConvNet(in_chans=1,mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 4, 2],dims=[32, 64, 192, 384], drop_path_rate=0.) 
        # mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2],dims=[32, 64, 192, 384]
        # self.convNet = ConvNet(in_channels=1)
        # self.graphNet = GraphNet(dim = 384, num_node=49, layer_depth=1)
        self.norm = nn.LayerNorm(128, eps=1e-6)
        self.classifier = nn.Linear(384, num_classes)
        
        # self.classifier_mid = nn.Linear(128, num_classes)
      

    def forward(self, x):
        # y1, y_mid = self.convNet(x)
        y1 = self.convNet(x)
        # print(y1.shape)
        # y2 = self.graphNet(y1)
        # print(y2.shape)
        # y2 = torch.mean(y2, dim=1)
        # y3 = self.classifier(y2)

        # y3 = F.adaptive_avg_pool2d(y2, 1)
        # y3 = y3.view(y3.size(0), -1)
        # y3 = self.classifier_mid(y3)

        return y1 #, pre_mid


if __name__ == '__main__':
    model = ConvGNN(3)
    x = torch.randn((4, 1, 224, 224))
    
    y = model(x)
    print(y)
    '''
    criterion = nn.CrossEntropyLoss()
    y1 = torch.rand((3, 3))
    yt = torch.tensor([1, 2, 0])
    print(y1)
    y_sum = y1[:,0]+y1[:,1]
    
    y2 = torch.stack([y_sum,y1[:,2]], dim = 1)
    y3 = torch.cat([y1[:,:2],torch.FloatTensor(3,1).fill_(0.)],dim=1)
    # print(len(y_sum))

    yt_2 = yt_3 = yt.clone()
    
    for i,label in enumerate(yt_2):
        if label != torch.tensor(2):
            yt_2[i] = 0
        else:
            yt_2[i] = 1
        
    print(yt_2)

    # for i,label in enumerate(yt_3):
    #     if label != torch.tensor(2):
    #         yt_3[i] = 0
    #     else:
    #         yt_3[i] = 1
    loss2 = criterion(y3,yt)
    print(yt)
    print(loss2)
    '''