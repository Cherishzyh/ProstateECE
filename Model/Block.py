import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = conv3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # 返回加权的 x
        # out = self.reasampler(x * psi)
        return x * psi


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(out_channels, out_channels, stride)

    def forward(self, x):

        residual = self.conv1(x)
        out = self.bn1(residual)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = residual + out
        out = self.relu2(out)
        # out = self.conv3(out)

        return out


# class ResidualBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out

# 3D
def conv3D3x3(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride, padding=1, bias=False)


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv3D, self).__init__()
        self.conv1 = conv3D3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm3d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3D3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm3d(filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv3d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.reasampler = nn.Sequential(
            nn.Conv3d(F_int, F_int, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(F_int)
        )


    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.psi(g1 + x1)
        # 返回加权的 x
        out = self.reasampler(x * psi)
        return out