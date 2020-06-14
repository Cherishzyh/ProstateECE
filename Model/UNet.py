import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = conv3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class UNet(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.up5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv6 = DoubleConv(512, 256)
        self.up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv7 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.conv9 = nn.Conv2d(64, out_channels, 1)
        self.fc1 = nn.Linear(1*184*184, 1*1000)
        self.fc2 = nn.Linear(1*1000, 1*2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        up_5 = self.up5(c4)
        merge6 = torch.cat((up_5, c3), dim=1)
        c5 = self.conv6(merge6)
        up_6 = self.up6(c5)
        merge7 = torch.cat((up_6, c2), dim=1)
        c7 = self.conv7(merge7)
        up_7 = self.up7(c7)
        merge8 = torch.cat((up_7, c1), dim=1)
        c8 = self.conv8(merge8)
        c9 = self.conv9(c8)
        c9 = c9.view(-1, 1*184*184)
        out = self.fc2(self.fc1(c9))
        # out = self.sigmoid(out)
        return out


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=3)
    model = model.to(device)
    print(model)


if __name__ == '__main__':

    test()