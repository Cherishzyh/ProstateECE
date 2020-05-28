import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from T4T.Utility.Data import *
from MeDIT.DataAugmentor import random_2d_augment

from Model.ResNet import ResNet, ResidualBlock
from DataSet.CheckPoint import EarlyStopping

from MeDIT.Normalize import Normalize01


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def Train():

    dataset = DataManager(random_2d_augment)

    dataset.AddOne(Image2D(r'X:\CNNFormatData\ProstateCancerECE\NPY\T2Slice\Train', shape=(184, 184)))
    dataset.AddOne(Image2D(r'X:\CNNFormatData\ProstateCancerECE\NPY\DwiSlice\Train', shape=(184, 184)))
    dataset.AddOne(Image2D(r'X:\CNNFormatData\ProstateCancerECE\NPY\AdcSlice\Train', shape=(184, 184)))
    dataset.AddOne(Feature(r'X:\CNNFormatData\ProstateCancerECE\NPY\csv\ece.csv'))
    dataset.AddOne(Image2D(r'X:\CNNFormatData\ProstateCancerECE\NPY\RoiSlice\Train', shape=(184, 184), is_roi=True), is_input=False)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=12, shuffle=True)

    model = ResNet(ResidualBlock, [1, 184, 184]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(150):
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, feature = inputs[0], inputs[1], inputs[2], inputs[3]
            inputs = torch.cat([t2, dwi, adc], axis=1)
            # inputs = inputs.to(device)
            # outputs = outputs.to(device)
            inputs = inputs.type(torch.FloatTensor).to(device)
            outputs = outputs.type(torch.FloatTensor).to(device)

            prediction = model(inputs)
            loss = criterion(prediction, outputs)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                template = r"Epoch:{}/{}, step:{}, Loss:{:.6f}"
                print(template.format(epoch + 1, 150, i + 1, loss.item()))

    # test model
    # model.eval()
    # with torch.no_grad():
    #     total = 0
    #     correct = 0
    #     for x, y in testloader:
    #         x = x.to(device)
    #         y = y.to(device)
    #         prediction = model(x)
    #         _, predic = torch.max(prediction.data, dim=1)
    #         total += y.size(0)
    #         correct += (predic == y).sum().item()
    #
    #     print("Accuracy:{}%".format(100 * correct / total))

    # save model
    torch.save(model.state_dict(), "resnet.ckpt")


if __name__ == '__main__':
    Train()