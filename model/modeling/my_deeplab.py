import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torchsummary import summary


"""
ResNet, Code is adapted from 
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



"""
Modified to support atrous convolution
"""
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding
       Dilated convolution should be pad with dilated quantity
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


"""
Modified to support atrous convolution
"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, dilation=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        #add dilation term
        self.conv2 = conv3x3(planes, planes, stride, groups, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2,
                 groups=1, width_per_group=64, norm_layer=None, pretrained=True):
        super(ResNet, self).__init__()

        #out stride default 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        muti_grid = [1,2,4]
        base_rate = 2

        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        #64 128 256 512
        self.inplanes = planes[0]
        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                               bias=False) #64 * h * w
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer)

        """
        Instead of original downsampling, atrous convolution is applied here
        """
        dilations = [x * base_rate for x in muti_grid]
        self.layer4 = self._make_dialted_layer(block, planes[3], layers[3], dilations, stride=1, groups=groups, norm_layer = norm_layer)
        #self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer)


        self._init_weight()
        if pretrained:
            self._load_pretrained_model()



        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677


    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)


    def _make_dialted_layer(self, block, planes, blocks, dilations, stride=1, groups=1, norm_layer=None):
        assert blocks == len(dilations)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, dilation=dilations[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, len(dilations)):
            rate = dilations[i]
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, dilation=rate))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x) #64*W/2*H/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)#256*W/4*H/4
        x = self.layer2(x)#512*W/8*H/8
        x = self.layer3(x)#1024*W/16*H/16
        x = self.layer4(x)#2048*W/16*H/16 since atrous convolution

        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
    #             groups=1, out_stride = 16, width_per_group=64, norm_layer=None):
    model = ResNet(**kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


"""
ASPP module
"""
class AtrousConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, dilation):
        super(AtrousConvBN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding, dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        """
        Perform multi atrous convolution with rate 1, 6, 12 18
        The padding will be 0, 6, 12, 18 in this case
        since rate 1 is simply kerner size 1
        Adapted from https://arxiv.org/pdf/1606.00915.pdf
                     https://arxiv.org/pdf/1706.05587.pdf
        """
        self.layer1 = AtrousConvBN(in_channels, out_channels, 1, 0, 1)
        self.layer2 = AtrousConvBN(in_channels, out_channels, 3, 6, 6)
        self.layer3 = AtrousConvBN(in_channels, out_channels, 3, 12, 12)
        self.layer4 = AtrousConvBN(in_channels, out_channels, 3, 18, 18)

        self.feature_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv = AtrousConvBN(out_channels*5, out_channels, 1, 0, 1)

    def forward(self, x):
        h, w = x.size()[2:]

        out1 = self.layer1(x) #256
        out2 = self.layer2(x) #256
        out3 = self.layer3(x) #256
        out4 = self.layer4(x) #256
        features = F.upsample(self.feature_pooling(x), size=(h, w), mode='bilinear') #256

        x = torch.cat([out1, out2, out3, out4, features], dim = 1)

        x = self.conv(x)

        return x

class My_DeepLab(nn.Module):
    def __init__(self, num_classes = 2, in_channels = 3):
        super(My_DeepLab, self).__init__()

        self.resnet = resnet101(pretrained=True, num_classes = num_classes, block = Bottleneck, layers= [3, 4, 23, 3])
        self.ASPP = ASPP(2048, 256)

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x):
            h, w = x.size()[2:]
            x = self.resnet(x)
            x = self.ASPP(x)
            x = self.decoder(x) #H/4 * W/4
            x = F.upsample(x, size=(h, w), mode='bilinear') #H*W
            return x





if __name__ == "__main__":
    model = My_DeepLab(2, 3)
    model.eval()
    image = torch.randn(1, 3, 600, 400)
    with torch.no_grad():
        output = model.forward(image)

    summary(model, (3, 600,400))



