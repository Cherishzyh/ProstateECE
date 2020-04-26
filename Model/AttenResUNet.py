import torch
import torch.nn as nn

from Model.Block import AttentionBlock, ResidualBlock, conv3x3


class AttenUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttenUNet, self).__init__()

        # Encoding Part
        # self.Conv1 = conv3x3(in_channels, 64)
        self.ResBlock1 = ResidualBlock(in_channels, 64)
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ResBlock2 = ResidualBlock(64, 128)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ResBlock3 = ResidualBlock(128, 256)
        self.Pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ResBlock4 = ResidualBlock(256, 512)

        # Decoding Part
        self.Up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=256, F_l=512, F_int=512)
        self.ResBlock5 = ResidualBlock(1024, 256)

        self.Up6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.Att6 = AttentionBlock(F_g=128, F_l=256, F_int=256)
        self.ResBlock6 = ResidualBlock(512, 128)

        self.Up7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.Att7 = AttentionBlock(F_g=64, F_l=128, F_int=128)
        self.ResBlock7 = ResidualBlock(256, 64)

        self.Conv1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmiod = nn.Sigmoid()

        self.ResBlock8 = ResidualBlock(512, 128)
        self.ResBlock9 = ResidualBlock(128, 32)
        self.ResBlock10 = ResidualBlock(32, 8)
        self.fc1 = nn.Linear(1*8*23*23, 1*1000)
        self.fc2 = nn.Linear(1*1000, 1*2)

    def forward(self, x):
        # encoding path
        # x = self.Conv1(x)
        res_1 = self.ResBlock1(x)        # 64*184*184
        pool_1 = self.Pool1(res_1)        # 64*92*92
        res_2 = self.ResBlock2(pool_1)        # 128*92*92
        pool_2 = self.Pool2(res_2)        # 128*46*46
        res_3 = self.ResBlock3(pool_2)        # 256*46*46
        pool_3 = self.Pool3(res_3)        # 256*23*23
        res_4 = self.ResBlock4(pool_3)        # 512*23*23

        # decoding path
        up_5 = self.Up5(res_4)         # 512*46*46
        atten_5 = self.Att5(g=res_3, x=up_5)        # 256*46*46 + 512*46*46 = 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)        # 512*46*46
        res_5 = self.ResBlock5(merge_5)        # 256*46*46

        up_6 = self.Up6(res_5)        # 256*92*92
        atten_6 = self.Att6(g=res_2, x=up_6)        # 128*92*92 + 256*92*92
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        res_6 = self.ResBlock6(merge_6)

        up_7 = self.Up7(res_6)
        atten_7 = self.Att7(g=res_1, x=up_7)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        res_7 = self.ResBlock7(merge_7)

        segmentation_out = self.Conv1x1(res_7)
        segmentation_out = self.sigmiod(segmentation_out)

        # ResNet
        res_8 = self.ResBlock8(res_4)
        res_9 = self.ResBlock9(res_8)
        res_10 = self.ResBlock10(res_9)
        fc = res_10.view(-1, 8*23*23)
        classification_out = self.fc2(self.fc1(fc))

        return segmentation_out, classification_out


def test():
    from tensorboardX import SummaryWriter
    x = torch.rand(12, 1, 184, 184)
    model = AttenUNet(in_channels=1, out_channels=1)
    print(model)
    with SummaryWriter(comment='AlexNet') as w:
        w.add_graph(model, x)



if __name__ == '__main__':
    test()