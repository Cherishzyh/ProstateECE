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

from CnnTools.T4T.Utility.Data import *
from BasicTool.MeDIT.Augment import *

from DataSet.CheckPoint import EarlyStopping

from CnnTools.T4T.Utility.Initial import HeWeightInit


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = r''


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/ct_slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/atten_slice', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/alltrain_label.csv'), is_input=False)
    data.Balance(Label(data_root + '/alltrain_label.csv'))

    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def Train():
    sub_train = []
    sub_val = []
    param_config = {
        RotateTransform.name: {'theta': ['uniform', -10, 10]},
        ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                              'vertical_shift': ['uniform', -0.05, 0.05]},
        ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                             'vertical_zoom': ['uniform', 0.95, 1.05]},
        FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
        BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                             'drop_ratio': ['uniform', 0., 1.]},
        NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
        ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
        GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
        ElasticTransform.name: ['elastic', 1, 0.1, 256]
    }
    input_shape = []
    batch_size = []

    train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(sub_val, param_config, input_shape, batch_size, True)

    torch.autograd.set_detect_anomaly(True)
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0.0
    valid_loss = 0.0
    cla_criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')
    model.apply(HeWeightInit)

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            ece = np.squeeze(outputs, axis=1)

            inputs = torch.cat([t2, dwi, adc, roi, prostate], dim=1)
            inputs = inputs.float().to(device)

            ece = ece.float().to(device)

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
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' % (epoch + 1, 1000, i + 1, train_loss / 10))
                print(list(class_out_sigmoid.cpu().detach().numpy()))
                train_loss = 0.0


        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(val_loader):
                t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
                ece = np.squeeze(outputs, axis=1)

                inputs = torch.cat([t2, dwi, adc, roi, prostate], dim=1)
                inputs = inputs.float().to(device)

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
                    print('Epoch [%d / %d], Iter [%d],  Valid Loss: %.4f' % (epoch + 1, 1000, i + 1, valid_loss / 10))
                    print(list(class_out_sigmoid.cpu().detach().numpy()))
                    valid_loss = 0.0

        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:')

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(sum(valid_loss_list)/len(valid_loss_list), model, save_path=model_folder, evaluation=min)

        if early_stopping.early_stop:
            print("Early stopping")
            break



if __name__ == '__main__':
    Train()
