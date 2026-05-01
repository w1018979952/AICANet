import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from GC_attention import ContextBlock
from torch.nn.utils import spectral_norm
from image_model import Resnet18
from res_se_34l_model import ResNetSE34

def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
class Generator(nn.Module):
    def __init__(self, d_conv_dim=32):
        super(Generator, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((3,3))
        self.ContextBlock = ContextBlock(512, 0.25) 
        self.generator = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(4608, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, True),
            snlinear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh()
            )
        self.apply(weights_init)        

    def forward(self, *input,mod='train'):
        Poutv1, Poutf1, inputv1,inputf1, VF1_Coef, Loss_Orthogonal1 = self.ContextBlock(input[0],input[1], mod='train')
        Poutv2, Poutf2, inputv2,inputf2, VF2_Coef, Loss_Orthogonal2 = self.ContextBlock(input[0],input[2], mod='train')
        Poutv = 0.5*(Poutv1 + Poutv2)
        input1v = 0.5*(inputv1 + inputv2)

        Pout_v = forward_feature(self, Poutv)
        Pout_f1 = forward_feature(self, Poutf1)
        Pout_f2 = forward_feature(self, Poutf2)
        
        outv1 = forward_feature(self, input1v)
        outf1 = forward_feature(self, inputf1)
        outf2 = forward_feature(self, inputf2)
        outv2 = forward_feature(self, input[3])
        
        Loss_total_Orthogonal = Loss_Orthogonal1 + Loss_Orthogonal2

        return Pout_v, Pout_f1, Pout_f1, outv1, outf1, outf2, VF1_Coef, VF2_Coef, Loss_total_Orthogonal

    def test_forward(self, *input,mod='train'):

        Poutv1, Poutf1, inputv1,inputf1, _, _ = self.ContextBlock(input[0],input[1], mod='train')
        Poutv2, Poutf2, inputv2,inputf2, _, _ = self.ContextBlock(input[0],input[2], mod='train')
        input1v = 0.5*(inputv1 + inputv2)

        outv1 = forward_feature(self, input1v)
        outf1 = forward_feature(self, inputf1)
        outf2 = forward_feature(self, inputf2)
        
        return outv1, outf1, outf2

def forward_feature(self,x): 
    N,C,_,_ = x.size()
    x_topk = self.gap(x)
    h3 = self.generator(x_topk.view(N,-1))
    
    return h3


class Dis(nn.Module):
    def __init__(self, d_conv_dim):
        super(Dis, self).__init__()
        self.Dtrans = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(in_features=d_conv_dim*4, out_features=2),
            nn.BatchNorm1d(2),
        )

    def forward(self, in1, in2):
        out1 = self.Dtrans(in1) 
        out2 = self.Dtrans(in2) 
        return out1, out2

class Class(nn.Module):
    def __init__(self, c_conv_dim):
        super(Class, self).__init__()
        self.trans = nn.Sequential(
            nn.Dropout(0.1),
            snlinear(in_features=c_conv_dim*4*2, out_features=2),
            nn.BatchNorm1d(2)
        )
        self.soft = nn.Softmax(dim=1)
    def forward(self, pinp1, pinp2, pinp3, inp1, inp2, inp3):
        in1 = torch.exp(inp1-inp2)
        in2 = torch.exp(inp1-inp3)
        out = torch.cat([in1, in2], dim=1)
        
        pin1 = torch.exp(pinp1-pinp2)
        pin2 = torch.exp(pinp1-pinp3)
        pout = torch.cat([pin1, pin2], dim=1)
        return self.trans(pout), self.trans(out)
        
    def test_forward(self,inp1, inp2, inp3):
        in1 = torch.exp(inp1-inp2)
        in2 = torch.exp(inp1-inp3)
        out = torch.cat([in1, in2], dim=1)

        return self.trans(out)

class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        self.feature_audio = ResNetSE34()
        self.feature_frame = Resnet18()


    def forward(self, v1, f1, f2, v2):
        v = self.feature_audio(v1)
        f1 = self.feature_frame(f1)
        f2 = self.feature_frame(f2)
        v2 = self.feature_audio(v2)
        return v, f1, f2, v2
        
    def test_forward(self, v1, f1, f2):
        v = self.feature_audio(v1)
        f1 = self.feature_frame(f1)
        f2 = self.feature_frame(f2)
        return v, f1, f2