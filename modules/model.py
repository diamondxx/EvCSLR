import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils.model_zoo as model_zoo
from torch.cuda.amp import autocast as autocast

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


class SelfAttention(nn.Module):
    """ Layer attention module"""
    def __init__(self, in_dim, down_rate, bias=True):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.down_rate = down_rate
        self.temperature = nn.Parameter(torch.ones(1))

        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x):
        """
            # x is [2, 64, 60, 14, 14]
            inputs :
                x : input feature maps(B C T H W)
            returns :
                out : attention value + input feature
                attention: (B C C)
        """
        m_batchsize, C, height, width = x.size()
        qkv = self.qkv_dwconv(self.qkv(x))      # [2, 64, 60, 56, 56]
        q, k, v = qkv.chunk(3, dim=1)   # [2, 1280, 56, 56]
        q = q.view(m_batchsize, C, -1)
        k = k.view(m_batchsize, C, -1)
        v = v.view(m_batchsize, C, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        self_attn = (q @ k.transpose(-2, -1)) * self.temperature    # [2, 64, 64]
        self_attn = self_attn.softmax(dim=-1)

        out_1 = (self_attn @ v)
        out_1 = out_1.view(m_batchsize, C, height, width)

        out_1 = self.project_out(out_1)

        out = out_1 + x
        return out
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAttention(nn.Module):
    """ MCAI module in EvCSLR paper"""

    def __init__(self, in_dim, down_rate, bias=True):
        super(CBAttention, self).__init__()
        self.chanel_in = in_dim
        self.down_rate = down_rate

        self.temperature = nn.Parameter(torch.ones(1))
        self.temperature_e1 = nn.Parameter(torch.ones(1))
        self.qkv = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_e1 = nn.Conv2d( self.chanel_in ,  self.chanel_in *3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.qkv_dwconv_e1 = nn.Conv2d(self.chanel_in*3, self.chanel_in*3, kernel_size=3, stride=1, padding=1, groups=self.chanel_in*3, bias=bias)
        self.project_out = nn.Conv2d(self.chanel_in, self.chanel_in, kernel_size=1, bias=bias)

    def forward(self, x, e1):
        """
            # x is [2, 64, 60, 14, 14],       e1 is [2, 64, 60, 14, 14]
            inputs :
                x : input feature maps(B C T H W)
            returns :
                out : attention value + input feature
                attention: (B C C)
        """
        m_batchsize, C, height, width = x.size()
        qkv = self.qkv_dwconv(self.qkv(x))      # [2, 64, 60, 56, 56]
        qkv_e1 = self.qkv_dwconv_e1(self.qkv_e1(e1))
        q, k, v = qkv.chunk(3, dim=1)   # [2, 1280, 56, 56]
        q = q.view(m_batchsize, C, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        q_e1, k_e1, v_e1 = qkv_e1.chunk(3, dim=1)
        k_e1 = k_e1.view(m_batchsize, C, -1)
        v_e1 = v_e1.view(m_batchsize, C, -1)
        k_e1 = torch.nn.functional.normalize(k_e1, dim=-1)
        attn_e1 = (q @ k_e1.transpose(-2, -1)) * self.temperature_e1    # [2, 64, 64]
        attn_e1 = attn_e1.softmax(dim=-1)
        out_1 = (attn_e1 @ v_e1)
        out_1 = out_1.view(m_batchsize, C, height, width)
        out_1 = self.project_out(out_1)
        out = out_1 + x
        return out

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1) -> None:
        super(ResBlock, self).__init__()
       
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)
 
        self.downsample = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
        
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.downsample(x)
        out = self.relu(out)
        return out


class STPF(nn.Module):
    """TRCorr module in EvCSLR paper"""
    def __init__(self, input_size):
        super(STPF, self).__init__()
        hidden_size = input_size//16
        self.conv_transform = nn.Conv1d(input_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.conv_back = nn.Conv1d(hidden_size, input_size, kernel_size=1, stride=1, padding=0)
        self.num = 3
        self.conv_enhance = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=int(i+1), groups=hidden_size, dilation=int(i+1)) for i in range(self.num)
        ])
        self.weights = nn.Parameter(torch.ones(self.num) / self.num, requires_grad=True)
        self.w = nn.Parameter(torch.ones(1), requires_grad=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, r_e2):
        out = self.conv_transform(x.mean(-1).mean(-1))
        aggregated_out = out
        for i in range(self.num):
            out = self.conv_enhance[i](out)
            aggregated_out = aggregated_out + self.weights[i] * out
        out = self.conv_back(aggregated_out)
        return r_e2 * F.sigmoid(out.unsqueeze(-1).unsqueeze(-1)) * self.w
 

class ResNet18(nn.Module):
    def __init__(self, ResBlock, num_classes=1000) -> None:
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_e1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_e2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_e1 = nn.BatchNorm2d(64)
        self.bn1_e2 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer1_e1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer1_e2 = self.make_layer(ResBlock, 64, 2, stride=1,sta="end")
        

        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer2_e1 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer2_e2 = self.make_layer(ResBlock, 128, 2, stride=2,sta="end")
        
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer3_e1 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer3_e2 = self.make_layer(ResBlock, 256, 2, stride=2,sta="end")
        
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.layer4_e1 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.layer4_e2 = self.make_layer(ResBlock, 512, 2, stride=2,sta="end")
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)


        self.self_atten_1 = SelfAttention(64, down_rate=2)
        self.self_atten_2 = SelfAttention(128, down_rate=4)
        self.self_atten_3 = SelfAttention(256, down_rate=8)
        self.self_atten_4 = SelfAttention(512, down_rate=8)
       
        # MCAI module
        self.nn_attn_e1_1 = CBAttention(64, down_rate=2)
        self.nn_attn_e1_2 = CBAttention(128, down_rate=4)
        self.nn_attn_e1_3 = CBAttention(256, down_rate=8)
        self.nn_attn_e1_4 = CBAttention(512, down_rate=8)
            
        self.nn_attn_e2_1 = CBAttention(64, down_rate=2)
        self.nn_attn_e2_2 = CBAttention(128, down_rate=4)
        self.nn_attn_e2_3 = CBAttention(256, down_rate=8)
        self.nn_attn_e2_4 = CBAttention(512, down_rate=8)
        
        self.att1 = ChannelAttention(64)
        self.att2 = ChannelAttention(64)
        self.att3 = ChannelAttention(64)
        self.att4 = ChannelAttention(512)
        
        # TRCorr module
        self.stpf1_1 = STPF(64)
        self.stpf1_2 = STPF(64)
        self.stpf1_3 = STPF(64)

        self.stpf2_1 = STPF(128)
        self.stpf2_2 = STPF(128)
        self.stpf2_3 = STPF(128)

        self.stpf3_1 = STPF(256)
        self.stpf3_2 = STPF(256)
        self.stpf3_3 = STPF(256)
        
        self.stpf4_1 = STPF(512)
        self.stpf4_2 = STPF(512)
        self.stpf4_3 = STPF(512)
        
    def forward(self, e1, x, e2, temp):
        # Here e1 is rgb, x is voxel grid, and e2 is event summation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = out * self.att1(out)
        
        out = self.maxpool(out)
      
        e1 = self.conv1_e1(e1)
        e1 = self.bn1_e1(e1)
        e1 = self.relu(e1)
        e1 = e1 * self.att2(e1)
        e1 = self.maxpool(e1)

        e2 = self.conv1_e2(e2)
        e2 = self.bn1_e2(e2)
        e2 = self.relu(e2)
        e2 = e2 * self.att3(e2)
        e2 = self.maxpool(e2)
        
        out = self.layer1(out)
        
        e1 = self.layer1_e1(e1)
        e2 = self.layer1_e2(e2)
        B,C,H,W = e1.shape
       
        e1_t = e1.reshape(B//temp,C,temp,H,W)
        out_t = out.reshape(B//temp,C,temp,H,W)
        e2_t = e2.reshape(B//temp,C,temp,H,W)
        e1 = e1 + self.stpf1_1(out_t, e1_t).reshape(e1.shape)
        out = out + self.stpf1_2(out_t,out_t).reshape(e1.shape)
        e2 = e2 + self.stpf1_3(e2_t,e2_t).reshape(e1.shape)
        
        out = self.self_atten_1(out)
        e1 = self.nn_attn_e1_1(out, e1)
        e2 = self.nn_attn_e2_1(out, e2)
        
        out = self.layer2(out)
        e1 = self.layer2_e1(e1)
        e2 = self.layer2_e2(e2)
        B,C,H,W = e1.shape
        
        e1_t = e1.reshape(B//temp,C,temp,H,W)
        out_t = out.reshape(B//temp,C,temp,H,W)
        e2_t = e2.reshape(B//temp,C,temp,H,W)
        e1 = e1 + self.stpf2_1(out_t, e1_t).reshape(e1.shape)
        out = out + self.stpf2_2(out_t,out_t).reshape(e1.shape)
        e2 = e2 + self.stpf2_3(e2_t,e2_t).reshape(e1.shape)
        out = self.self_atten_2(out)
        e1 =  self.nn_attn_e1_2(out, e1)
        e2 = self.nn_attn_e2_2(out, e2)
        
        out = self.layer3(out)
        e1 = self.layer3_e1(e1)
        e2 = self.layer3_e2(e2)
        B,C,H,W = e1.shape
        e1_t = e1.reshape(B//temp,C,temp,H,W)
        out_t = out.reshape(B//temp,C,temp,H,W)
        e2_t = e2.reshape(B//temp,C,temp,H,W)
        e1 = e1 + self.stpf3_1(out_t, e1_t).reshape(e1.shape)
        out = out + self.stpf3_2(out_t,out_t).reshape(e1.shape)
        e2 = e2 + self.stpf3_3(e2_t,e2_t).reshape(e1.shape)
       
        out = self.self_atten_3(out)
        e1 = self.nn_attn_e1_3(out, e1)
        e2 = self.nn_attn_e2_3(out, e2)

        out = self.layer4(out)
        e1 = self.layer4_e1(e1)
        e2 = self.layer4_e2(e2)
        B,C,H,W = e1.shape
        e1_t = e1.reshape(B//temp,C,temp,H,W)
        out_t = out.reshape(B//temp,C,temp,H,W)
        e2_t = e2.reshape(B//temp,C,temp,H,W)
        e1 = e1 + self.stpf4_1(out_t, e1_t).reshape(e1.shape)
        out = out + self.stpf4_2(out_t,out_t).reshape(e1.shape)
        e2 = e2 + self.stpf4_3(e2_t,e2_t).reshape(e1.shape)
        
        out = self.self_atten_4(out)
        e1 = self.nn_attn_e1_4(out, e1)
        e2 = self.nn_attn_e2_4(out, e2)
        
        out = e1 + out + e2
        out = out * self.att4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
 
    def make_layer(self, block, channels, num_blocks, stride,sta=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        tmp = self.inchannel
        for stride in strides:
            layers.append(block(tmp, channels, stride))
            tmp = channels
        if sta == "end":
            self.inchannel = channels
        return nn.Sequential(*layers)
    

def resnet18():
    model = ResNet18(ResBlock)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    
    keys = list(checkpoint.keys())
    for i in keys:
        if "fc" not in i:
            e1 = i.replace(i.split(".")[0],i.split(".")[0] + "_e1")
            e2 = i.replace(i.split(".")[0],i.split(".")[0] + "_e2")
            checkpoint[e1] = checkpoint[i]
            checkpoint[e2] = checkpoint[i]
    model.load_state_dict(checkpoint, strict=False)
    return model
