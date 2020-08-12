import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil

from sklearn import metrics

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

from Metric.classification_statistics import get_auc, draw_roc
from Metric.MyMetric import BinaryClassification
from DataSet.MyDataLoader import LoadTVData, LoadTestData


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)

def Train():
    train_loader, validation_loader = LoadTVData()
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0.0
    valid_loss = 0.0
    cla_criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            ece = np.squeeze(outputs, axis=1)

            inputs = torch.cat([t2, dwi, adc, roi, prostate], axis=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            class_out, _ = model(inputs)
            class_out = torch.squeeze(class_out, dim=1)
            class_out_sigmoid = class_out.sigmoid()

            loss = cla_criterion(class_out, ece)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute auc
            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out_sigmoid.cpu().detach().numpy()))

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' %(epoch + 1, 1000, i + 1, train_loss / 10))
                print(list(class_out_sigmoid.cpu().detach().numpy()))
                train_loss = 0.0

        _, _, train_auc = get_auc(class_pred_list, class_list)
        class_list, class_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
                ece = np.squeeze(outputs, axis=1)

                inputs = torch.cat([t2, dwi, adc, roi, prostate], axis=1)
                inputs = inputs.type(torch.FloatTensor).to(device)

                ece = ece.type(torch.FloatTensor).to(device)

                class_out, _ = model(inputs)
                class_out = torch.squeeze(class_out, dim=1)
                class_out_sigmoid = class_out.sigmoid()

                loss = cla_criterion(class_out, ece)

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                # compute auc
                class_list.extend(list(ece.cpu().numpy()))
                class_pred_list.extend(list(class_out_sigmoid.cpu().detach().numpy()))

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d],  Valid Loss: %.4f' %(epoch + 1, 1000, i + 1, valid_loss / 10))
                    print(list(class_out_sigmoid.cpu().detach().numpy()))
                    valid_loss = 0.0
            _, _, valid_auc = get_auc(class_pred_list, class_list)

        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.add_scalars('Train_Val_auc',
                           {'train_auc': train_auc, 'val_auc': valid_auc}, epoch + 1)
        writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:',
              np.mean(valid_loss_list), 'Train auc:', train_auc, 'Valid auc:', valid_auc)

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(sum(valid_loss_list)/len(valid_loss_list), model, save_path=model_folder, evaluation=min)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    test_loader = LoadTestData(data_folder)
    train_loader, validation_loader = LoadTVData(data_folder, is_test=True)

    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))

    fpr_list, tpr_list, auc_list = [], [], []

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [train_loader, validation_loader, test_loader]

    # with torch.no_grad():
    model.eval()
    for name_num, loader in enumerate(loader_list):
        class_list, class_pred_list = [], []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            ece = np.squeeze(outputs, axis=1)

            inputs = torch.cat([t2, dwi, adc, roi, prostate], dim=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.FloatTensor).to(device)

            class_out = model(inputs)
            class_out = torch.squeeze(class_out, dim=1)
            class_out_sigmoid = class_out.sigmoid()

            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out_sigmoid.cpu().detach().numpy()))

        fpr, tpr, auc = get_auc(class_pred_list, class_list)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    draw_roc(fpr_list, tpr_list, auc_list, name_list)


def ComputeMetric(loader, model, is_metric=False):
    model.eval()
    class_list, class_pred_list = [], []

    for i, (inputs, outputs) in enumerate(loader):
        t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        ece = np.squeeze(outputs, axis=1)

        inputs = torch.cat([t2, dwi, adc, roi, prostate], dim=1)
        # inputs = torch.cat([t2, dwi, adc], dim=1)
        inputs = inputs.type(torch.FloatTensor).to(device)

        ece = ece.type(torch.FloatTensor).to(device)

        class_out = model(inputs)

        class_out = torch.squeeze(class_out, dim=1)
        class_out_sigmoid = class_out.sigmoid()

        class_list.append(int(ece.cpu().numpy()))
        class_pred_list.append(float(class_out_sigmoid.cpu().detach().numpy()))
        print(i)

    if is_metric:
        metric = BinaryClassification()
        metric_dict = metric.Run(class_pred_list, class_list)
        print(metric_dict)
    return class_pred_list, class_list



if __name__ == '__main__':
    # Train()
    # Test()
    # FeatureMap()

    data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY'

    model_folder = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNet505Inputs'
    model_path = os.path.join(model_folder, 'checkpoint.pt')
    graph_path = os.path.join(model_folder, 'logs')

    # from MyModel.ResNetcbam import ResNet, Bottleneck
    from MyModel.ResNet50 import ResNet, Bottleneck
    loader = LoadTestData(data_folder)
    train_loader, val_loader = LoadTVData(data_folder, is_test=True)

    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))
    # ComputeMetric(val_loader, model, is_metric=True)
    Test()