import torch
import torch.nn as nn

from Model.Block import *


class AttenUNetMultiTask2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttenUNetMultiTask2D, self).__init__()
        self.EncodingResBlock1 = ResidualBlock(in_channels, 64)
        self.EncodingPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.EncodingResBlock2 = ResidualBlock(64, 128)
        self.EncodingPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.EncodingResBlock3 = ResidualBlock(128, 256)
        self.EncodingPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.EncodingResBlock4 = ResidualBlock(256, 512)

        # Decoding Part
        self.DecodingUp5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.DecodingAtt5 = AttentionBlock(F_g=256, F_l=512, F_int=512)
        self.DecodingResBlock5 = ResidualBlock(1024, 256)

        self.DecodingUp6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.DecodingAtt6 = AttentionBlock(F_g=128, F_l=256, F_int=256)
        self.DecodingResBlock6 = ResidualBlock(512, 128)

        self.DecodingUp7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.DecodingAtt7 = AttentionBlock(F_g=64, F_l=128, F_int=128)
        self.DecodingResBlock7 = ResidualBlock(256, 64)

        self.DecodingConv1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        # self.Decodingsigmiod = nn.Sigmoid()

        self.ClassifiedResBlock8 = ResidualBlock(512, 128)
        self.ClassifiedResBlock9 = ResidualBlock(128, 32)
        self.ClassifiedResBlock10 = ResidualBlock(32, 8)
        self.ClassifiedFc1 = nn.Linear(1 * 8 * 23 * 23, 1 * 1000)
        self.ClassifiedFc2 = nn.Linear(1 * 1000, 1 * 2)
        self.Classifiedsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # encoding path
        # x = self.Conv1(x)
        res_1 = self.EncodingResBlock1(x)        # 64*184*184
        pool_1 = self.EncodingPool1(res_1)        # 64*92*92
        res_2 = self.EncodingResBlock2(pool_1)        # 128*92*92
        pool_2 = self.EncodingPool2(res_2)        # 128*46*46
        res_3 = self.EncodingResBlock3(pool_2)        # 256*46*46
        pool_3 = self.EncodingPool3(res_3)        # 256*23*23
        res_4 = self.EncodingResBlock4(pool_3)        # 512*23*23

        # decoding path
        up_5 = self.DecodingUp5(res_4)         # 512*46*46
        atten_5 = self.DecodingAtt5(g=res_3, x=up_5)        # 256*46*46 + 512*46*46 = 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)        # 512*46*46
        res_5 = self.DecodingResBlock5(merge_5)        # 256*46*46

        up_6 = self.DecodingUp6(res_5)        # 256*92*92
        atten_6 = self.DecodingAtt6(g=res_2, x=up_6)        # 128*92*92 + 256*92*92
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        res_6 = self.DecodingResBlock6(merge_6)

        up_7 = self.DecodingUp7(res_6)
        atten_7 = self.DecodingAtt7(g=res_1, x=up_7)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        res_7 = self.DecodingResBlock7(merge_7)

        segmentation = self.DecodingConv1x1(res_7)
        # segmentation_out = self.Decodingsigmiod(segmentation)

        # ResNet
        res_8 = self.ClassifiedResBlock8(res_4)
        res_9 = self.ClassifiedResBlock9(res_8)
        res_10 = self.ClassifiedResBlock10(res_9)
        fc = res_10.view(-1, 8*23*23)
        classification = self.ClassifiedFc2(self.ClassifiedFc1(fc))
        classification = self.Classifiedsoftmax(classification)

        return segmentation, classification


class AttenUNetMultiTask3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttenUNetMultiTask3D, self).__init__()
        self.EncodingResBlock1 = DoubleConv3D(in_channels, 64)
        self.EncodingPool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.EncodingResBlock2 = DoubleConv3D(64, 128)
        self.EncodingPool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.EncodingResBlock3 = DoubleConv3D(128, 256)
        self.EncodingPool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.EncodingResBlock4 = DoubleConv3D(256, 512)

        # Decoding Part
        self.DecodingUp5 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.DecodingAtt5 = AttentionBlock3D(F_g=512, F_l=256, F_int=512)
        self.DecodingResBlock5 = DoubleConv3D(768, 256)

        self.DecodingUp6 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.DecodingAtt6 = AttentionBlock3D(F_g=256, F_l=128, F_int=256)
        self.DecodingResBlock6 = DoubleConv3D(384, 128)

        self.DecodingUp7 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.DecodingAtt7 = AttentionBlock3D(F_g=128, F_l=64, F_int=128)
        self.DecodingResBlock7 = DoubleConv3D(192, 64)

        self.DecodingConv1x1 = nn.Conv3d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.Decodingsigmiod = nn.Sigmoid()

        self.ClassifiedResBlock8 = DoubleConv3D(512, 128)
        self.ClassifiedResBlock9 = DoubleConv3D(128, 32)
        self.ClassifiedResBlock10 = DoubleConv3D(32, 8)
        self.ClassifiedFc1 = nn.Linear(1 * 8 * 23 * 23, 1 * 1000)
        self.ClassifiedFc2 = nn.Linear(1 * 1000, 1 * 2)
        self.Classifiedsigmiod = nn.Sigmoid()

    def forward(self, x):
        # encoding path
        # x = self.Conv1(x)
        res_1 = self.EncodingResBlock1(x)        # 64*184*184
        pool_1 = self.EncodingPool1(res_1)        # 64*92*92
        res_2 = self.EncodingResBlock2(pool_1)        # 128*92*92
        pool_2 = self.EncodingPool2(res_2)        # 128*46*46
        res_3 = self.EncodingResBlock3(pool_2)        # 256*46*46
        pool_3 = self.EncodingPool3(res_3)        # 256*23*23
        res_4 = self.EncodingResBlock4(pool_3)        # 512*23*23

        # decoding path
        up_5 = self.DecodingUp5(res_4)         # 512*46*46
        atten_5 = self.DecodingAtt5(g=up_5, x=res_3)        # 256*46*46 + 512*46*46 = 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)        # 512*46*46
        res_5 = self.DecodingResBlock5(merge_5)        # 256*46*46

        up_6 = self.DecodingUp6(res_5)        # 256*92*92
        atten_6 = self.DecodingAtt6(g=up_6, x=res_2)        # 128*92*92 + 256*92*92
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        res_6 = self.DecodingResBlock6(merge_6)

        up_7 = self.DecodingUp7(res_6)
        atten_7 = self.DecodingAtt7(g=up_7, x=res_1)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        res_7 = self.DecodingResBlock7(merge_7)

        segmentation_out = self.DecodingConv1x1(res_7)
        segmentation_out = self.Decodingsigmiod(segmentation_out)

        # ResNet
        res_8 = self.ClassifiedResBlock8(res_4)
        res_9 = self.ClassifiedResBlock9(res_8)
        res_10 = self.ClassifiedResBlock10(res_9)
        fc = res_10.view(-1, 8*23*23)
        classification_out = self.ClassifiedFc2(self.ClassifiedFc1(fc))
        classification_out = self.Classifiedsigmiod(classification_out)

        return segmentation_out, classification_out
        # return classification_out


def test():

    model = AttenUNetMultiTask2D(in_channels=3, out_channels=1)
    # print(model)

    # from tensorboardX import SummaryWriter
    # x = torch.rand(12, 1, 184, 184)
    # with SummaryWriter(comment='AlexNet') as w:
    #     w.add_graph(model, x)

    paras = dict(model.named_parameters())
    # for k, v in paras.items():
    #     print(k.ljust(40), str(v.shape).ljust(30), 'bias:', v.requires_grad)
    encoding_paras = []
    decoding_paras = []
    classified_paras = []
    for k, v in paras.items():
        if 'Encoding' in k:
            encoding_paras.append(v)
        elif 'Decoding' in k:
            decoding_paras.append(v)
        else:
            classified_paras.append(v)

    print(encoding_paras)
    print(decoding_paras)
    print(classified_paras)

    params = [
        {"params": encoding_paras, "lr": 0.001},
        {"params": decoding_paras, "lr": 0.002},
        {"params": classified_paras, "lr": 0.003}
    ]
    optimizer = torch.optim.Adam(params)
    print(optimizer)


if __name__ == '__main__':
    test()