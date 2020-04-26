import torch
import torch.nn as nn

from Model.Block import DoubleConv, AttentionBlock


class AttenUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttenUNet, self).__init__()

        self.Conv1 = DoubleConv(in_channels, 64)
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = DoubleConv(64, 128)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = DoubleConv(128, 256)
        self.Pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = DoubleConv(256, 512)

        self.Up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=256, F_l=512, F_int=512)
        self.Conv5 = DoubleConv(1024, 256)

        self.Up6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.Att6 = AttentionBlock(F_g=128, F_l=256, F_int=256)
        self.Conv6 = DoubleConv(512, 128)

        self.Up7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.Att7 = AttentionBlock(F_g=64, F_l=128, F_int=128)
        self.Conv7 = DoubleConv(256, 64)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmiod = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        conv_1 = self.Conv1(x)        # 64*184*184
        pool_1 = self.Pool1(conv_1)        # 64*92*92
        conv_2 = self.Conv2(pool_1)        # 128*92*92
        pool_2 = self.Pool2(conv_2)        # 128*46*46
        conv_3 = self.Conv3(pool_2)        # 256*46*46
        pool_3 = self.Pool3(conv_3)        # 256*23*23
        conv_4 = self.Conv4(pool_3)        # 512*23*23

        # encoding path
        up_5 = self.Up5(conv_4)         # 512*46*46
        atten_5 = self.Att5(g=conv_3, x=up_5)        # 256*46*46 + 512*46*46 = 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)        # 512*46*46
        conv_5 = self.Conv5(merge_5)        # 256*46*46

        up_6 = self.Up6(conv_5)        # 256*92*92
        atten_6 = self.Att6(g=conv_2, x=up_6)        # 128*92*92 + 256*92*92
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        conv_6 = self.Conv6(merge_6)

        up_7 = self.Up7(conv_6)
        atten_7 = self.Att7(g=conv_1, x=up_7)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        conv_7 = self.Conv7(merge_7)

        out = self.Conv_1x1(conv_7)
        out = self.sigmiod(out)

        return out


def test():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AttenUNet(in_channels=1, out_channels=1)
    # model = model.to(device)
    print(model)


if __name__ == '__main__':
    test()