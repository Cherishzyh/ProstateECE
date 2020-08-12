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

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.BasicTool.MeDIT.Augment import config_example
from T4T.Utility.Loss import DiceLoss, CrossEntropy
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01

from DataSet.CheckPoint import EarlyStopping
from DataSet.MyDataLoader import LoadTVData, LoadTestData
from MyModel.SENet_154 import ResNeXt

from NPYFilePath import *
from Metric.classification_statistics import get_auc, draw_roc
from Metric.MyMetric import BinaryClassification

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY'
model_folder = r'/home/zhangyihong/Documents/ProstateECE/Model/DistanceMap/SeResNext4Inputs'
model_path = r'/home/zhangyihong/Documents/ProstateECE/Model/DistanceMap/SeResNext4Inputs/checkpoint.pt'
graph_path = r'/home/zhangyihong/Documents/ProstateECE/Model/DistanceMap/SeResNext4Inputs/logs'

def LoadTVData(folder, is_test=False, setname=None):
    if setname is None:
        setname = ['Train', 'Validation']

    t2_folder = os.path.join(folder, 'T2Slice')
    dwi_folder = os.path.join(folder, 'DwiSlice')
    adc_folder = os.path.join(folder, 'AdcSlice')
    distance_folder = os.path.join(folder, 'DistanceMap')

    csv_folder = os.path.join(folder, 'csv')
    ece_folder = os.path.join(csv_folder, 'ece.csv')

    train_t2_folder = os.path.join(t2_folder, setname[0])
    train_dwi_folder = os.path.join(dwi_folder, setname[0])
    train_adc_folder = os.path.join(adc_folder, setname[0])
    train_distance_folder = os.path.join(distance_folder, setname[0])

    validation_t2_folder = os.path.join(t2_folder, setname[1])
    validation_dwi_folder = os.path.join(dwi_folder, setname[1])
    validation_adc_folder = os.path.join(adc_folder, setname[1])
    validation_distance_folder = os.path.join(distance_folder, setname[1])


    if is_test:
        train_dataset = DataManager()
        validation_dataset = DataManager()
    else:
        train_dataset = DataManager(config_example)
        validation_dataset = DataManager(config_example)

    ###########################################################
    train_dataset.AddOne(Image2D(train_t2_folder, shape=(184, 184)))
    train_dataset.AddOne(Image2D(train_dwi_folder, shape=(184, 184)))
    train_dataset.AddOne(Image2D(train_adc_folder, shape=(184, 184)))
    train_dataset.AddOne(Image2D(train_distance_folder, shape=(184, 184), is_roi=True))

    train_dataset.AddOne(Feature(ece_folder), is_input=False)

    # train_dataset.Balance(Label(ece_folder))

    ###########################################################
    validation_dataset.AddOne(Image2D(validation_t2_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_dwi_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_adc_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_distance_folder, shape=(184, 184), is_roi=True))

    validation_dataset.AddOne(Feature(ece_folder), is_input=False)

    ###########################################################
    if is_test:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=True)

    return train_loader, validation_loader


def LoadTestData(folder):
    t2_folder = os.path.join(folder, 'T2Slice')
    dwi_folder = os.path.join(folder, 'DwiSlice')
    adc_folder = os.path.join(folder, 'AdcSlice')
    distance_folder = os.path.join(folder, 'DistanceMap')

    csv_folder = os.path.join(folder, 'csv')
    ece_folder = os.path.join(csv_folder, 'ece.csv')

    test_t2_folder = os.path.join(t2_folder, 'Test')
    test_dwi_folder = os.path.join(dwi_folder, 'Test')
    test_adc_folder = os.path.join(adc_folder, 'Test')
    test_distance_folder = os.path.join(distance_folder, 'Test')

    test_dataset = DataManager()

    ###########################################################
    test_dataset.AddOne(Image2D(test_t2_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(test_dwi_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(test_adc_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(test_distance_folder, shape=(184, 184), is_roi=True))

    test_dataset.AddOne(Feature(ece_folder), is_input=False)

    ###########################################################

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def Train():
    # ClearGraphPath(graph_path)
    train_loader, validation_loader = LoadTVData(data_folder)
    model = ResNeXt(4, 32, [3, 4, 6, 3], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3]
            ece = outputs.to(device)

            inputs = torch.cat([t2, dwi, adc, dismap], dim=1)
            inputs = inputs.float().to(device)

            class_out = model(inputs)
            class_out_sigmoid = class_out.sigmoid()

            loss = criterion(class_out, ece)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute auc
            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out_sigmoid.cpu().detach().numpy()))

            if (i + 1) % 5 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' % (epoch + 1, 1000, i + 1, train_loss / 5))
                # print(list(class_out_sigmoid.cpu().detach().numpy()))
                train_loss = 0.0

        _, _, train_auc = get_auc(class_pred_list, class_list)
        class_list, class_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3]
                ece = outputs.to(device)

                inputs = torch.cat([t2, dwi, adc, dismap], dim=1)
                inputs = inputs.float().to(device)

                class_out = model(inputs)
                class_out_sigmoid = class_out.sigmoid()

                loss = criterion(class_out, ece)

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                # compute auc
                class_list.extend(list(ece.cpu().numpy()))
                class_pred_list.extend(list(class_out_sigmoid.cpu().detach().numpy()))

                if (i + 1) % 5 == 0:
                    print('Epoch [%d / %d], Iter [%d],  Valid Loss: %.4f' %(epoch + 1, 1000, i + 1, valid_loss / 5))
                    # print(list(class_out_sigmoid.cpu().detach().numpy()))
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

    model = ResNeXt(4, 32, [3, 4, 6, 3], 1).to(device)
    model.load_state_dict(torch.load(model_path))

    fpr_list, tpr_list, auc_list = [], [], []

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [train_loader, validation_loader, test_loader]

    # with torch.no_grad():
    model.eval()
    for name_num, loader in enumerate(loader_list):
        class_list, class_pred_list = [], []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3]
            ece = outputs.to(device)

            inputs = torch.cat([t2, dwi, adc, dismap], dim=1)
            inputs = inputs.float().to(device)

            class_out = model(inputs)
            class_out_sigmoid = class_out.sigmoid()

            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out_sigmoid.cpu().detach().numpy()))

        fpr, tpr, auc = get_auc(class_pred_list, class_list)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    draw_roc(fpr_list, tpr_list, auc_list, name_list)


def ComputePcaMetric(loader, model, device, is_metric=False):
    model.eval()
    pca_true_list, pca_pred_list = [], []

    for i, (inputs, outputs) in enumerate(loader):
        t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3]
        ece = outputs.to(device)

        inputs = torch.cat([t2, dwi, adc, dismap], dim=1)
        inputs = inputs.float().to(device)

        class_out = model(inputs)
        class_out_sigmoid = class_out.sigmoid()

        pca_true_list.append(int(ece.cpu().numpy()))
        pca_pred_list.append(float(class_out_sigmoid.cpu().detach().numpy()))
        print(i)
    if is_metric:
        metric = BinaryClassification()
        metric_dict = metric.Run(pca_pred_list, pca_true_list)
        print(metric_dict)
    return pca_pred_list, pca_true_list

if __name__ == '__main__':
    # Train()
    # train_loader, val_loader = LoadTVData(data_folder, is_test=True)
    # test_loader = LoadTestData(data_folder)
    # model = ResNeXt(4, 32, [3, 4, 6, 3], 1).to(device)
    # model.load_state_dict(torch.load(model_path))
    # ece_pred_list, ece_true_list = ComputePcaMetric(train_loader, model, device, is_metric=True)
    Test()