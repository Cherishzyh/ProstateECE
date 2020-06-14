import torch
import torch.nn as nn

from Model.Block import DoubleConv
from Model.ResNet50 import ResNet, Bottleneck, BasicBlock


class MultiTaskModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiTaskModel, self).__init__()

        # Encoding Path
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)

        # Decoding Path 1
        self.up5a = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6a = DoubleConv(512, 256)
        self.up6a = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7a = DoubleConv(256, 128)
        self.up7a = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8a = DoubleConv(128, 64)
        self.conv9a = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

        # Decoding Path 2
        self.up5b = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6b = DoubleConv(512, 256)
        self.up6b = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7b = DoubleConv(256, 128)
        self.up7b = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8b = DoubleConv(128, 64)
        self.conv9b = nn.Conv2d(64, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

        # ResNet
        self.resnet34 = ResNet(BasicBlock, [3, 4, 6, 3])

    def forward(self, x):
        # Encoding path
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)

        # Decoding Path 1
        up_5a = self.up5a(c4)
        merge6a = torch.cat((up_5a, c3), dim=1)
        c5a = self.conv6a(merge6a)
        up_6a = self.up6a(c5a)
        merge7a = torch.cat((up_6a, c2), dim=1)
        c7a = self.conv7a(merge7a)
        up_7a = self.up7a(c7a)
        merge8a = torch.cat((up_7a, c1), dim=1)
        c8a = self.conv8a(merge8a)
        c9a = self.conv9a(c8a)
        outa = self.sigmoid(c9a)

        # Decoding Path 2
        up_5b = self.up5b(c4)
        merge6b = torch.cat((up_5b, c3), dim=1)
        c5b = self.conv6b(merge6b)
        up_6b = self.up6b(c5b)
        merge7b = torch.cat((up_6b, c2), dim=1)
        c7b = self.conv7b(merge7b)
        up_7b = self.up7b(c7b)
        merge8b = torch.cat((up_7b, c1), dim=1)
        c8b = self.conv8b(merge8b)
        c9b = self.conv9b(c8b)
        outb = self.sigmoid(c9b)

        class_inputs = torch.cat((x, outa, outb), dim=1)
        outc = self.resnet34(class_inputs)

        return outa, outb, outc


if __name__ == '__main__':
    model = MultiTaskModel(in_channels=3, out_channels=1)
    print(model)