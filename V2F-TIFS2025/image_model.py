
# Author: David Harwath, Wei-Ning Hsu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class feature_only_F(nn.Module):
    def __init__(self):
        super(feature_only_F, self).__init__()
        self.Conv2d1=nn.Conv2d(3, 3, 7, 2, 3)
        self.AvgPool2d1=nn.AvgPool2d(kernel_size=5, stride=2)
        self.BatchNorm2d1=nn.BatchNorm2d(3)
        self.LeakyReLU1=nn.LeakyReLU(0.2, True)
        self.Conv2d2=nn.Conv2d(3, 32, 5, 2, 2)
        self.AvgPool2d2=nn.AvgPool2d(kernel_size=5, stride=2)
        self.BatchNorm2d2=nn.BatchNorm2d(32)
        self.LeakyReLU2=nn.LeakyReLU(0.2, True)
        self.Conv2d3=nn.Conv2d(32, 64, 3, 1, 4)
        self.BatchNorm2d3=nn.BatchNorm2d(64)
        self.LeakyReLU3=nn.LeakyReLU(0.2, True)
        self.Conv2d4=nn.Conv2d(64, 256, 3, 1, 1)
        self.BatchNorm2d4=nn.BatchNorm2d(256)
        self.Conv2d5=nn.Conv2d(256, 512, 3, 1, 1)
        self.BatchNorm2d5=nn.BatchNorm2d(512)
        #self.Self_Attn1= Self_Attn(512)
        self.LeakyReLU5=nn.LeakyReLU(0.2, True)
        self.Conv2d6=nn.Conv2d(512, 1024, 3, 1, 1)
        self.BatchNorm2d6=nn.BatchNorm2d(1024)
        self.LeakyReLU6=nn.LeakyReLU(0.2, True)
        self.AvgPool2d5=nn.AvgPool2d(kernel_size=17, stride=2)#, padding=1


    def forward(self, f1):
        return feature_frame(self, f1)

def feature_frame(self,x):
    #x=N*3*224*224
    h1=self.Conv2d1(x)#50, 3, 112, 112
    h1=self.AvgPool2d1(h1)#50, 3, 56, 56
    h1=self.BatchNorm2d1(h1)#50, 3, 56, 56
    h1=self.LeakyReLU1(h1)#50, 3, 56, 56
    h2=self.Conv2d2(h1)#50, 5, 28, 28 ##50, 6, 28, 28
    h2=self.AvgPool2d2(h2)#50, 5, 14, 14  #50, 6, 14, 14
    h2=self.BatchNorm2d2(h2)#50, 5, 14, 14 #50, 6, 14, 14
    h2=self.LeakyReLU2(h2)#50, 5, 14, 14 #50, 6, 14, 14
    h3=self.Conv2d3(h2)#50, 8, 21, 21 #50, 12, 21, 21
    h3=self.BatchNorm2d3(h3)#50, 8, 21, 21 #50, 12, 21, 21
    h3=self.LeakyReLU3(h3)#50, 8, 21, 21 #50, 12, 21, 21
    h4=self.Conv2d4(h3)#50, 16, 21, 21
    h4=self.BatchNorm2d4(h4)#50, 16, 21, 21
    h5=self.Conv2d5(h4)#50, 32, 21, 21
    #h5=self.AvgPool2d5(h5)#50, 32, 10, 10
    #h5=self.Self_Attn1(h5)#50, 32, 10, 10
    h5=self.BatchNorm2d5(h5)#50, 32, 21, 21
    h6=self.LeakyReLU5(h5)#50, 32, 12, 12
    h6=self.Conv2d4(h6)
    h6=self.BatchNorm2d4(h6)
    h6=self.LeakyReLU5(h6)
    h7=self.AvgPool2d5(h6)#50, 32, 10, 10

    return h6



class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=512, pretrained=True):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            #self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))
            self.loaded = torch.hub.load_state_dict_from_url(imagemodels.ResNet18_Weights.IMAGENET1K_V1.url)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Resnet34(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
            #self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
            self.loaded = torch.hub.load_state_dict_from_url(imagemodels.ResNet18_Weights.IMAGENET1K_V1.url)
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet50_1(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50_1, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class VGG16(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(VGG16, self).__init__()
        seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        seed_model = nn.Sequential(*list(seed_model.children())[:-1]) # remove final maxpool
        last_layer_index = len(list(seed_model.children()))
        seed_model.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        self.image_model = seed_model

    def forward(self, x):
        x = self.image_model(x)
        return x


class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [2, 2, 2, 2])#3463
        if pretrained:
            model_url = imagemodels.resnet.model_urls['resnet50']
            self.load_state_dict(model_zoo.load_url(model_url))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.output_layer = Sequential(BatchNorm2d(1024),
                            Dropout(),
                            Flatten(),
                            Linear(1024 * 7 * 7, 1024),
                            BatchNorm1d(1024))
        self.Maxpool = nn.MaxPool2d(kernel_size=(7,7), stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        x1 = self.output_layer(x)
        #x2 = x.reshape(x.size()[0],x.size()[1],x.size()[2]*x.size()[3])
        b, c, _, _ = x.size()
        #x = x.view(-1, c, h*w)
        x1 = x1.view(b, c, 1)
        x_avg = self.avg_pool(x).view(b, c, 1)
        x_max = self.Maxpool(x).view(b, c, 1)
        x = torch.cat((x1, x_avg, x_max),2)
        #x =torch.cat(x1,x2,x3)
        return x