import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from T4T.Utility.Data import *
from MeDIT.DataAugmentor import random_2d_augment
from T4T.Utility.Loss import DiceLoss, CrossEntropy
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01

from DataSet.CheckPoint import EarlyStopping

from MyModel.MultiTaskModel2 import MultiSegModel
from DataSet.MyDataLoader import LoadTVData, LoadTestData
from NPYFilePath import data_folder
from Metric.Loss import BCEFocalLoss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain'
model_folder = r'/home/zhangyihong/Documents/ProstateECE/Model/MultiSegmentation'
model_path = r'/home/zhangyihong/Documents/ProstateECE/Model/MultiSegmentation/checkpoint.pt'
graph_path = r'/home/zhangyihong/Documents/ProstateECE/Model/MultiSegmentation/logs'


def BinaryPred(prediction):
    one = torch.ones_like(prediction)
    zero = torch.zeros_like(prediction)
    binary_prediction = torch.where(prediction > 0.5, one, zero)
    return binary_prediction


def ClearGraphPath():
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def Train():
    ClearGraphPath()
    train_loader, validation_loader = LoadTVData(is_test=False, folder=data_folder, setname=['PreTrain', 'PreValid'])
    model = MultiSegModel(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    seg_criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss1_list, valid_loss1_list = [], []
        train_loss2_list, valid_loss2_list = [], []
        train_loss_list, valid_loss_list = [], []
        train_loss1 = 0.0
        train_loss2 = 0.0
        train_loss = 0.0
        valid_loss1 = 0.0
        valid_loss2 = 0.0
        valid_loss = 0.0

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc = inputs[0], inputs[1], inputs[2],
            roi, prostate = outputs[0].to(device), outputs[1].to(device)

            inputs = torch.cat([t2, dwi, adc], dim=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            roi_out, prostate_out = model(inputs)

            loss1 = seg_criterion(roi_out, roi)
            loss2 = seg_criterion(prostate_out, prostate)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss1 += loss1.item()
            train_loss1_list.append(loss1.item())

            train_loss2 += loss2.item()
            train_loss2_list.append(loss2.item())

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Cancer train Loss: %.4f, Prostate train Loss: %.4f, Loss: %.4f' %
                      (epoch + 1, 1000, i + 1, train_loss1 / 10, train_loss2 / 10,  train_loss / 10))
                train_loss = 0.0
                train_loss1 = 0.0
                train_loss2 = 0.0

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc = inputs[0], inputs[1], inputs[2],
                roi, prostate = outputs[0].to(device), outputs[1].to(device)

                inputs = torch.cat([t2, dwi, adc], dim=1)
                inputs = inputs.type(torch.FloatTensor).to(device)


                roi_out, prostate_out = model(inputs)

                loss1 = seg_criterion(roi_out, roi)
                loss2 = seg_criterion(prostate_out, prostate)
                loss = loss1 + loss2

                valid_loss1 += loss1.item()
                valid_loss1_list.append(loss1.item())

                valid_loss2 += loss2.item()
                valid_loss2_list.append(loss2.item())

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d], Cancer validation Loss: %.4f, Prostate validation Loss: %.4f, Loss: %.4f' %
                          (epoch + 1, 1000, i + 1, valid_loss1 / 10, valid_loss2 / 10, valid_loss / 10))
                    valid_loss1 = 0.0
                    valid_loss2 = 0.0
                    valid_loss = 0.0

        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Train_Val_Loss1',
                           {'train_cancer_dice_loss': np.mean(train_loss1_list), 'val_cancer_dice_loss': np.mean(valid_loss1_list)}, epoch + 1)
        writer.add_scalars('Train_Val_Loss2',
                           {'train_prostate_dice_loss': np.mean(train_loss2_list), 'val_prostate_dice_loss': np.mean(valid_loss2_list)}, epoch + 1)
        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.close()

        # print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list))

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(sum(valid_loss_list)/len(valid_loss_list), model, save_path=model_folder, evaluation=min)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    from Metric.Dice import Dice
    import matplotlib.pyplot as plt
    dice = Dice()
    test_loader = LoadTestData(is_test=True)
    train_loader, validation_loader = LoadTVData(is_test=True)

    model = MultiSegModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [train_loader, validation_loader, test_loader]

    # with torch.no_grad():
    model.eval()
    for name_num, loader in enumerate(loader_list):
        class_list, class_pred_list = [], []
        prostate_list, prostate_pred_list = [], []
        roi_list, roi_pred_list = [], []
        prostate_dice, roi_dice = [], []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc = inputs[0], inputs[1], inputs[2],
            roi, prostate = outputs[0].to(device), outputs[1].to(device)

            inputs = torch.cat([t2, dwi, adc], dim=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            roi_out, prostate_out = model(inputs)

            prostate_out = BinaryPred(prostate_out).cpu().detach()
            prostate = prostate.cpu()
            prostate_pred_list.extend(list(prostate_out))
            prostate_list.extend(list(prostate))

            roi_out = BinaryPred(roi_out).cpu().detach()
            roi = roi.cpu()
            roi_pred_list.extend(list(roi_out))
            roi_list.extend(list(roi))

        for idx in range(len(roi_list)):
            roi_dice.append(dice(roi_list[idx], roi_pred_list[idx]).numpy())
            prostate_dice.append(dice(prostate_list[idx], prostate_pred_list[idx]).numpy())
        print('average dice of roi in', name_list[name_num], ':', sum(roi_dice)/len(roi_dice))
        print('average dice of prostate in', name_list[name_num], ':', sum(prostate_dice) / len(prostate_dice))
        plt.hist(roi_dice)
        plt.title('Dice of Prostate Cancer in ' + name_list[name_num])
        plt.show()

        plt.hist(prostate_dice)
        plt.title('Dice of Prostate in ' + name_list[name_num])
        plt.show()


def ShowPicture():
    from Metric.Dice import Dice

    dice = Dice()
    train_loader, validation_loader = LoadTVData(is_test=True)
    test_loader = LoadTestData(is_test=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MultiSegModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    ece_pre_list = []
    ece_list = []
    for i, (inputs, outputs) in enumerate(test_loader):
        t2, dwi, adc = inputs[0], inputs[1], inputs[2],
        roi, prostate = outputs[0].to(device), outputs[1].to(device)

        inputs = torch.cat([t2, dwi, adc], axis=1)
        inputs = inputs.type(torch.FloatTensor).to(device)

        roi_out, prostate_out = model(inputs)


        # prostate_out = np.squeeze(BinaryPred(prostate_out).cpu().detach().numpy())
        # prostate = np.squeeze(prostate.cpu().numpy())
        # roi_out = np.squeeze(BinaryPred(roi_out).cpu().detach().numpy())
        # roi = np.squeeze(roi.cpu().numpy())
        # t2_data = np.squeeze(t2.cpu().numpy())
        # plt.imshow(t2_data, cmap='gray')
        # plt.contour(prostate, colors='r')
        # plt.contour(prostate_out, colors='y')
        # plt.contour(roi, colors='g')
        # plt.contour(roi_out, colors='b')
        # plt.title('ECE: ' + str(ece.cpu().numpy()[0][0])
        #           + ', predict ece: ' + str(class_out.cpu().detach().numpy()[0][0]))
        # plt.axis('off')
        # plt.show()
    plt.hist(ece_list)
    plt.show()


def FeatureMap():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MultiSegModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))

    adc = np.load(os.path.join(test_adc_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    dwi = np.load(os.path.join(test_dwi_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    t2 = np.load(os.path.join(test_t2_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    prostate = np.load(os.path.join(test_prostate_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    cancer = np.load(os.path.join(test_roi_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))

    inputs = np.concatenate([t2, dwi, adc], axis=0)
    inputs = inputs[np.newaxis, :]
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(device)

    _, _, x, feature_map = model(inputs)

    plt.subplot(331)
    plt.title('t2')
    plt.imshow(np.squeeze(t2), cmap='gray')
    plt.contour(np.squeeze(prostate), colors='y')
    plt.contour(np.squeeze(cancer), colors='r')
    plt.axis('off')

    plt.subplot(332)
    plt.title('dwi')
    plt.imshow(np.squeeze(dwi), cmap='gray')
    plt.contour(np.squeeze(cancer), colors='r')
    plt.axis('off')

    plt.subplot(333)
    plt.title('adc')
    plt.imshow(np.squeeze(adc), cmap='gray')
    plt.contour(np.squeeze(cancer), colors='r')
    plt.axis('off')

    plt.subplot(334)
    plt.title('conv1')
    plt.imshow(feature_map['conv1'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(335)
    plt.title('layer1')
    plt.imshow(feature_map['layer1'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(336)
    plt.title('layer2')
    plt.imshow(feature_map['layer2'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(337)
    plt.title('layer3')
    plt.imshow(feature_map['layer3'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.subplot(338)
    plt.title('layer4')
    plt.imshow(feature_map['layer4'][0, 0, ...].cpu().detach().numpy(), cmap='gray')
    # plt.contour(prostate, colors='y')
    # plt.contour(cancer, colors='r')
    plt.axis('off')

    plt.show()





if __name__ == '__main__':
    Train()
    # Test()
    # ShowPicture()
    # FeatureMap()