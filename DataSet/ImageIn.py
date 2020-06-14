import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from pytorchtools import EarlyStopping

import matplotlib.pyplot as plt

from T4T.Utility.Loader import ImageInImageOutDataSet, BinaryOneImageOneLabelTest
from T4T.Utility.ImageProcessor import ImageProcess2D
from MeDIT.DataAugmentor import random_2d_augment

from Metric.Metric import Dice
from Metric.classification_statistics import get_auc, compute_confusion_matrix
from Model.AttenUnet import AttenUNet

from FilePath import *
from DataSet.CheckPoint import EarlyStopping


def BinaryPred(prediction):
    one = torch.ones_like(prediction)
    zero = torch.zeros_like(prediction)
    binary_prediction = torch.where(prediction > 0.5, one, zero)
    return binary_prediction


def Train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)

    data_shape = {'input_0': (1, 184, 184), 'output_0': (1, 184, 184)}
    not_roi_info = {'input_0': True, 'output_0': False}

    writer = SummaryWriter(log_dir=graph_folder, comment='Net')

    train_set = ImageInImageOutDataSet(root_folder=train_folder,
                                       data_shape=data_shape,
                                       not_roi_info=not_roi_info,
                                       processor=processor)

    valid_set = ImageInImageOutDataSet(root_folder=validation_folder,
                                       not_roi_info=not_roi_info,
                                       data_shape=data_shape,
                                       processor=processor)

    train_loader = DataLoader(train_set, batch_size=12, shuffle=True)

    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True)

    model = UNet(in_channels=1, out_channels=1)
    model = model.to(device)

    lr = 0.001
    train_loss = 0.0
    valid_loss = 0.0
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    dice = Dice()
    for epoch in range(150):
        train_loss_list, valid_loss_list= [], []
        train_dice_list, valid_dice_list = [], []
        label_list, pred_list = [], []

        model.train()
        for i, train_data in enumerate(train_loader):
            inputs, labels = train_data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            outputs = BinaryPred(outputs)
            label_list.extend(list(torch.squeeze(labels)))
            pred_list.extend(list(torch.squeeze(outputs)))
            train_dice = 0
            for index in range(len(label_list)):
                train_dice += dice.forward(pred_list[index], label_list[index])
                # plt.subplot(121)
                # plt.imshow(label_list[index].cpu().detach().numpy(), cmap='gray')
                # plt.subplot(122)
                # plt.imshow(pred_list[index].cpu().detach().numpy(), cmap='gray')
                # plt.show()
            train_dice = train_dice / len(label_list)

            train_loss_list.append(loss.item())
            train_dice_list.append(train_dice)
            train_loss += loss.item()

            if (i + 1) % 1 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f, Train Dice: %.4f' %
                      (epoch + 1, 150, i + 1, train_loss/10, sum(train_dice_list)/len(train_dice_list)))
                train_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i, valid_data in enumerate(valid_loader):
                inputs, labels = valid_data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                outputs = BinaryPred(outputs)
                valid_loss_list.append(loss.item())
                valid_loss += loss.item()

                label_list.extend(list(torch.squeeze(labels)))
                pred_list.extend(list(torch.squeeze(outputs)))

                valid_dice = 0
                for index in range(len(label_list)):
                    valid_dice += dice.forward(pred_list[index], label_list[index])
                valid_dice = valid_dice / len(label_list)

                valid_dice_list.append(valid_dice)
                valid_loss_list.append(loss.item())

                if (i + 1) % 1 == 0:
                    print('Epoch [%d / %d], Iter [%d], Valid Loss: %.4f, valid Dice: %.4f' %
                          (epoch + 1, 150, i + 1, valid_loss/10, sum(valid_dice_list)/len(valid_dice_list)))
                    valid_loss = 0.0

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.add_scalars('Train_Val_Dice',
                           {'train_auc': torch.mean(torch.stack(train_dice_list)), 'val_auc': torch.mean(torch.stack(valid_dice_list))}, epoch + 1)
        writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list))

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(np.mean(valid_loss_list), model, save_path=model_save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

def Test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_shape = {'input_0': (1, 184, 184), 'output_0': (1, 184, 184)}
    not_roi_info = {'input_0': True, 'output_0': False}

    # processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)
    test_set = ImageInImageOutDataSet(root_folder=test_folder,
                                    data_shape=data_shape,
                                    not_roi_info=not_roi_info)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # model = torch.load(model_path)
    model = AttenUNet(in_channels=1, out_channels=1)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    dice = Dice()
    test_dice = []
    np_dice = []
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs, labels = test_data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = BinaryPred(outputs)

            test_dice.append(dice.forward(torch.squeeze(outputs), torch.squeeze(labels)))
            plt.subplot(121)
            plt.imshow(torch.squeeze(inputs).cpu().detach().numpy(), cmap='gray')
            plt.contour(torch.squeeze(labels).cpu().detach().numpy(), colors='r')
            plt.subplot(122)
            plt.imshow(torch.squeeze(outputs).cpu().detach().numpy(), cmap='gray')
            plt.savefig(os.path.join(image_folder, str(i)+'.jpg'))
            plt.close()

        print(sum(test_dice)/len(test_dice))
        for index in range(len(test_dice)):
            np_dice.append(test_dice[index].cpu().detach().numpy())
        plt.hist(np_dice)
        plt.show()


if __name__ == '__main__':
    # data_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0/Train'
    # case_list = os.listdir(data_path)
    # print(case_list)
    Train()
    # Test()
