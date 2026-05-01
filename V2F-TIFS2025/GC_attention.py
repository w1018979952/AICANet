import torch
# from mmcv.cnn import constant_init, kaiming_init
from torch import nn
from torch.nn import init
# from torch.nn.init import constant_init
import torch.nn.functional as F
from CBilinearPooling import CompactBilinearPooling
from torch.nn.utils import spectral_norm
def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)



class ModalNorm(nn.Module):
    def __init__(self, dim=128):
        super(ModalNorm, self).__init__()
        self.gama = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cur_mean = x.mean(dim =[2,3], keepdim=True)
        cur_var = x.var(dim =[2,3], keepdim=True)
        x_hat = (x - cur_mean) / torch.sqrt(cur_var + 1e-6)
        gamma = self.conv_gamma(x)
        beta = self.conv_beta(x)
        f = self.gama * x_hat + self.beta
        return f

class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes # 1024
        self.ratio = ratio #1024 *(1/4-1/16)
        self.lamb = 0.2
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        self.CBPooling = CompactBilinearPooling(128,128,4000)
        self.LayerNorm = ModalNorm()
        self.softmax  = nn.Softmax(dim=-1)
        self.MAM = MAM(self.inplanes)

        self.Shrinkage = Shrinkage(self.planes, gap_size = (self.planes,1))
        self.LeakyReLU = nn.LeakyReLU(0.2, True)
        self.gap = nn.AdaptiveMaxPool2d((3,3))#nn.AdaptiveAvgPool2d((3,3))
        
        self.Avgap = nn.AdaptiveAvgPool2d(1)
        
        self.ReLU = nn.ReLU(inplace=True)
        self.IN = nn.InstanceNorm2d(inplanes, track_running_stats=False)
        self.down_dim = nn.Conv2d(self.inplanes, self.planes, kernel_size=1,bias=False)
        
        self.Share_Afeature = snconv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=1, stride=1, padding=0,bias=False)
        self.Share_Vfeature = snconv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=1, stride=1, padding=0,bias=False)

        self.Private_Afeature = snconv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=1, stride=1, padding=0,bias=False)
        self.Private_Vfeature = snconv2d(in_channels=self.inplanes, out_channels=self.planes, kernel_size=1, stride=1, padding=0,bias=False)
        self.Pricate_Adecode = nn.Sequential(nn.Conv2d(self.planes, self.inplanes, kernel_size=1,bias=False),
                                      nn.Sigmoid())
        self.Pricate_Vdecode = nn.Sequential(nn.Conv2d(self.planes, self.inplanes, kernel_size=1,bias=False),
                                      nn.Sigmoid())

        self.cosine = nn.CosineEmbeddingLoss()
        
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1,bias=False),
                nn.Sigmoid()
                )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                self.LayerNorm,
                nn.ReLU(inplace=True), 
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

        self.estimation_Coeff = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(4000, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
            nn.Tanh()
            )

    def reset_parameters(self):
        if self.pooling_type == 'att':
            self.conv_mask.apply(weights_init_kaiming)
            self.conv_mask.inited = True

    def spatial_pool(self, input_y, x):
        batch, channel, height, width = x.size()
        input_y = input_y.view(batch, channel, height * width)
        input_y = input_y.unsqueeze(1)
        if self.pooling_type == 'att':
            input_x = x.view(batch, channel, height * width)#x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, C, H*W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, H * W, 1]
            context = torch.matmul(input_y, context_mask)# [N, 1, C, HW] [N, 1, HW, 1]= [N, 1, C, 1]
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context
        
    def Self_Attention(self, Q,K,V, use_DropKey=True, mask_ratio=0.1, Class_Drop='Drop'):
        attn = Q@ K.transpose(-2, -1) 
        if use_DropKey == True and Class_Drop == 'Drop':
            m_r = torch.ones_like(attn) * mask_ratio
            att = attn + torch.bernoulli(m_r) * -1e-12
            attn = att.softmax(dim=-1)
            
        elif use_DropKey == True and Class_Drop == 'Shrinkage':
            attn = 10*self.Shrinkage(attn)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        x= attn @ V
        
        return x
        
    def Cross_Attention(self, Q,K,V, ECoef, use_DropKey=True, mask_ratio=0.1, Class_Drop='Drop'):
        attn = Q@ K.transpose(-2, -1)
        ECoef_Inter = torch.sign(ECoef.expand_as(attn))
        New_att = ECoef_Inter*attn
        
        if use_DropKey == True and Class_Drop == 'Drop':        
            m_r = torch.ones_like(New_att) * mask_ratio
            att = New_att + torch.bernoulli(m_r) * -1e-12
            attn = att.softmax(dim=-1)
            
        elif use_DropKey == True and Class_Drop == 'Shrinkage':
            attn = 10*self.Shrinkage(New_att)
            attn = attn.softmax(dim=-1)
        else:
            attn = New_att.softmax(dim=-1)
        x= attn @ V
        
        return x

    def forward(self, x, x2, mod='test'):
        # [N, C, 1, 1]
        N, C, h, w = x.size()
        _, _, h1, w1 = x2.size()

        contextA = self.Share_Afeature(x)#.view(N,self.planes,-1)#B*C*hw
        context_A = contextA.view(N,self.planes,-1)#B*C*hw
        context2V = self.Share_Vfeature(x2)#.view(N,self.planes,-1)#B*C*hw
        context_2V = context2V.view(N,self.planes,-1)#B*C*hw
        
        context_PA = self.Private_Afeature(x).view(N,self.planes,-1)
        context_PV = self.Private_Vfeature(x2).view(N,self.planes,-1)
        
        contextPA = context_PA / torch.norm(context_PA, 2, 0, True) # 1024, N1
        contextPV = context_PV / torch.norm(context_PV, 2, 0, True) # 1024, N2
        
        PcontextA = self.gap(contextA).view(N,self.planes,-1)#torch.topk(context_A, k=9, dim=2, largest=True)[0]
        Pcontext2V = self.gap(context2V).view(N,self.planes,-1)#torch.topk(context_2V, k=9, dim=2, largest=True)[0]
        
        context = PcontextA / torch.norm(PcontextA, 2, 0, True) # 1024, N1
        context2 = Pcontext2V / torch.norm(Pcontext2V, 2, 0, True) # 1024, N2

        Att_cross = self.CBPooling(context, context2)#B*8000
        ECoefficient = self.estimation_Coeff(Att_cross)
        ECoef = ECoefficient.unsqueeze(1)
        
        if mod == 'train':
            context_PA_cross = self.Self_Attention(contextPA, contextPA, context_PA, True, 0.1, 'Shrinkage').view(N,self.planes, h, w)
            context_PV_cross = self.Self_Attention(contextPV, contextPV, context_PV, True, 0.1, 'Shrinkage').view(N,self.planes, h1, w1)
        
            #share_Cross_inter
            Cross_inter_A = self.Cross_Attention(context, context2, context_A, ECoef, True, 0.1, 'Shrinkage')
            Cross_inter_V = self.Cross_Attention(context2, context, context_2V, ECoef, True, 0.1, 'Shrinkage')
        
            Label_context_cross = self.Self_Attention(context_A, Cross_inter_A, context_A, False, 0.0, 'Shrinkage').view(N,self.planes, h, w)
            Label_context2_cross = self.Self_Attention(context_2V, Cross_inter_V, context_2V, False, 0.0, 'Shrinkage').view(N,self.planes, h1, w1)
        elif mod == 'test':
            context_PA_cross = self.Self_Attention(contextPA, contextPA, context_PA, True, 0.0, 'Shrinkage').view(N,self.planes, h, w)
            context_PV_cross = self.Self_Attention(contextPV, contextPV, context_PV, True, 0.0, 'Shrinkage').view(N,self.planes, h1, w1)
        
            #share_Cross_inter
            Cross_inter_A = self.Cross_Attention(context, context2, context_A, ECoef, True, 0.0, 'Shrinkage')
            Cross_inter_V = self.Cross_Attention(context2, context, context_2V, ECoef, True, 0.0, 'Shrinkage')
        
            Label_context_cross = self.Self_Attention(context_A, Cross_inter_A, context_A, False, 0.0, 'Shrinkage').view(N,self.planes, h, w)
            Label_context2_cross = self.Self_Attention(context_2V, Cross_inter_V, context_2V, False, 0.0, 'Shrinkage').view(N,self.planes, h1, w1)
        else:       
            context_PA_cross = self.Self_Attention(contextPA, contextPA, context_PA, False, 0.1, 'Drop').view(N,self.planes, h, w)
            context_PV_cross = self.Self_Attention(contextPV, contextPV, context_PV, False, 0.1, 'Drop').view(N,self.planes, h1, w1)
        
            #share_Cross_inter
            Cross_inter_A = self.Cross_Attention(context, context2, context_A, ECoef, False, 0.1, 'Drop')
            Cross_inter_V = self.Cross_Attention(context2, context, context_2V, ECoef, False, 0.1, 'Drop')
        
            Label_context_cross = self.Self_Attention(context_A, Cross_inter_A, context_A, False, 0.0, 'Shrinkage').view(N,self.planes, h, w)
            Label_context2_cross = self.Self_Attention(context_2V, Cross_inter_V, context_2V, False, 0.0, 'Shrinkage').view(N,self.planes, h1, w1)
        
    
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(Label_context_cross)#.view(N,-1).unsqueeze(2)
            channel_add_term2 = self.channel_add_conv(Label_context2_cross)#.view(N,-1).unsqueeze(2)
            
            
            Pricate_A = self.Pricate_Adecode(context_PA_cross)
            Pricate_V = self.Pricate_Vdecode(context_PV_cross)
            PoutA = Pricate_A + x
            PoutV = Pricate_V + x2
            
            out = self.MAM(x, channel_add_term, Pricate_A) 
            out2 = self.MAM(x2, channel_add_term2, Pricate_V)

            Loss_Orthogonal = self.cosine(self.Avgap(channel_add_term).squeeze(), self.Avgap(Pricate_A).squeeze(),torch.tensor([-1]).cuda()).mean(0) + \
                    self.cosine(self.Avgap(channel_add_term2).squeeze(), self.Avgap(Pricate_V).squeeze(),torch.tensor([-1]).cuda()).mean(0)
           
        return PoutA, PoutV, out, out2, ECoefficient, Loss_Orthogonal
        
class MAM(nn.Module):
    def __init__(self, dim, r=4):
        super(MAM, self).__init__()
        self.channel_attention = nn.Sequential(
                nn.Conv2d(dim, dim // r, kernel_size=1, bias=False), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)
        # self.IMN = InstanceModalNorm(dim)
    def forward(self, out, x, x_private):
        Avg_pooled = F.avg_pool2d(x, x.size()[2:])
        Max_pooled = F.max_pool2d(x, x.size()[2:])
        
        Avg_private = F.avg_pool2d(x_private, x_private.size()[2:])
        Max_private = F.max_pool2d(x_private, x_private.size()[2:])
        
        Avg_mask = self.channel_attention(Avg_pooled)
        Max_mask = self.channel_attention(Max_pooled)
        mask = Avg_mask + Max_mask
        
        Avg_private_mask = self.channel_attention(Avg_private)
        Max_private_mask = self.channel_attention(Max_private)
        mask_private = Avg_private_mask + Max_private_mask
        
        out = out * (1- mask - mask_private) + self.IN(x) * mask + self.IN(x_private) * mask_private
        return out     

class Shrinkage(nn.Module):
    def __init__(self,  channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(2*channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )


    def forward(self, x):
     
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        average = self.gap(x).squeeze()
        x1= torch.max(x,dim=1)[0]
        x2= torch.max(x,dim=2)[0]
        x = torch.cat([x1,x2],dim=1)
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).expand_as(x_raw)
        
        # soft thresholding
        sub = torch.where(x_abs < x, x-x, x_abs)
        x = torch.mul(torch.sign(x_raw), sub)#.squeeze(-1)
        return x 