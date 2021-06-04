from __future__ import division
import math
import torch.nn as nn
import torch.nn.functional as F

import torch

from MyModel.Block import conv1x1, conv3x3
from T4T.Block.ConvBlock import ConvBn2D


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

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


class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, baseWidth=4, cardinality=32,
                 stride=1, downsample=None, downstride=2):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(Bottleneck, self).__init__()

        conv_planes = planes // 2

        D = int(math.floor(conv_planes * (baseWidth / 16)))
        C = cardinality

        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, conv_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(conv_planes)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(conv_planes)
        self.sa_fm = SpatialAttention()

        if downsample is True:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, conv_planes * 2, downstride),
                nn.BatchNorm2d(conv_planes * 2),
            )
        else:
            self.downsample = None

    def forward(self, feature_map, dis_map):

        residual = feature_map

        out = self.conv1(feature_map)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shape = out.shape[2:]
        dis_map_resize = F.interpolate(dis_map, size=shape, mode='bilinear', align_corners=True)

        out = self.ca(out) * out
        out_fm = self.sa_fm(out) * out
        out_dm = dis_map_resize * out

        out = torch.cat([out_fm, out_dm], dim=1)

        if self.downsample is not None:
            residual = self.downsample(feature_map)

        out += residual
        out = self.relu(out)

        return out


class Layer1(nn.Module):
    def __init__(self, inplanes, outplanes, cardinality=32):
        super(Layer1, self).__init__()
        self.layer1_0 = Bottleneck(inplanes=inplanes, planes=outplanes, cardinality=cardinality,
                                   downsample=True, downstride=1)
        self.layer1_1 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        self.layer1_2 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)

    def forward(self, x, dis_map):
        x = self.layer1_0(x, dis_map)
        x = self.layer1_1(x, dis_map)
        x = self.layer1_2(x, dis_map)
        return x


class Layer2(nn.Module):
    def __init__(self, inplanes, outplanes, cardinality=32):
        self.inplanes = inplanes

        super(Layer2, self).__init__()
        self.layer2_0 = Bottleneck(inplanes=inplanes, planes=outplanes, cardinality=cardinality,
                                   stride=2, downsample=True)
        self.layer2_1 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        self.layer2_2 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        self.layer2_3 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)

    def forward(self, x, dis_map):
        x = self.layer2_0(x, dis_map)
        x = self.layer2_1(x, dis_map)
        x = self.layer2_2(x, dis_map)
        x = self.layer2_3(x, dis_map)

        return x


class Layer3(nn.Module):
    def __init__(self, inplanes, outplanes, cardinality=32):
        super(Layer3, self).__init__()
        self.layer3_0 = Bottleneck(inplanes=inplanes, planes=outplanes, cardinality=cardinality,
                                   stride=2, downsample=True)
        self.layer3_1 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        self.layer3_2 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        self.layer3_3 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        # self.layer3_4 = Bottleneck(inplanes=self.inplanes*4, planes=self.inplanes//2, cardinality=cardinality)
        # self.layer3_5 = Bottleneck(inplanes=self.inplanes*4, planes=self.inplanes//2, cardinality=cardinality)

    def forward(self, x, dis_map):
        x = self.layer3_0(x, dis_map)
        x = self.layer3_1(x, dis_map)
        x = self.layer3_2(x, dis_map)
        x = self.layer3_3(x, dis_map)
        # x = self.layer3_4(x, dis_map)
        # x = self.layer3_5(x, dis_map)
        return x


class Layer4(nn.Module):
    def __init__(self, inplanes, outplanes, cardinality=32):
        super(Layer4, self).__init__()
        self.layer4_0 = Bottleneck(inplanes=inplanes, planes=outplanes, cardinality=cardinality,
                                   stride=2, downsample=True)
        self.layer4_1 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)
        self.layer4_2 = Bottleneck(inplanes=outplanes, planes=outplanes, cardinality=cardinality)

    def forward(self, x, dis_map):
        x = self.layer4_0(x, dis_map)
        x = self.layer4_1(x, dis_map)
        x = self.layer4_2(x, dis_map)

        return x


class ResNeXt(nn.Module):
    def __init__(self, in_channels, num_classes, num_feature, inplanes=32):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResNeXt, self).__init__()

        self.conv1 = ConvBn2D(in_channels, inplanes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = Layer1(inplanes, inplanes * 2, cardinality=8)
        self.layer2 = Layer2(inplanes * 2, inplanes * 4, cardinality=16)
        self.layer3 = Layer3(inplanes * 4, inplanes * 6, cardinality=24)
        self.layer4 = Layer4(inplanes * 6, inplanes * 8, cardinality=32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(inplanes * 8, inplanes),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(inplanes, num_classes)
        self.fc3 = nn.Linear(num_classes+num_feature, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, t2, adc, dwi, dis_map, feature):
        inputs = torch.cat([t2, adc, dwi], dim=1)
        x = self.conv1(inputs)
        x = self.maxpool1(x)  # shape = (92, 92)

        x = self.layer1(x, dis_map)  # shape = (92, 92)
        x = self.layer2(x, dis_map)  # shape = (46, 46)
        x = self.layer3(x, dis_map)  # shape = (23, 23)
        x = self.layer4(x, dis_map)  # shape = (12, 12)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x1 = self.fc2(x)
        x = torch.cat((x1, feature), dim=1)
        x2 = self.fc3(x)
        return torch.softmax(x1, dim=1), torch.softmax(x2, dim=1)



if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResNeXt(3, num_classes=2, num_feature=5).to(device)
    print(model)
    inputs = torch.randn(1, 1, 184, 184).to(device)
    dis_map = torch.randn(1, 1, 184, 184).to(device)
    feature = torch.randn(1, 5).to(device)
    prediction = model(inputs, inputs, inputs, dis_map, feature)
    print(prediction[0].shape, prediction[1].shape)