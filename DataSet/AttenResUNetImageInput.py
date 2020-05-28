import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
# from pytorchtools import EarlyStopping

from T4T.Utility.Loader import ImageInImageOutDataSet
from T4T.Utility.ImageProcessor import ImageProcess2D
from MeDIT.DataAugmentor import random_2d_augment

from Metric.classification_statistics import get_auc, compute_confusion_matrix
from Model.UNet import Unet
from FilePath import *
from DataSet.CheckPoint import EarlyStopping


def Train():
    graph_folder = r''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)

    data_shape = {'input_0': (1, 184, 184), 'input_1': (1, 184, 184), 'input_2': (1, 184, 184),
                  'output_0': (1, 184, 184), 'output_1': (2,)}

    not_roi_info = {'input_0': True, 'input_1': True, 'input_2': True, 'output_0': False, 'output_1': False}
    #
    # data_shape = {'input_0': (1, 184, 184), 'input_1': (1, 184, 184), 'input_2': (1, 184, 184),
    #               'output_0': (1, 184, 184)}
    #
    # not_roi_info = {'input_0': True, 'input_1': True, 'input_2': True, 'output_0': False}
    # writer = SummaryWriter(log_dir=graph_folder, comment='Net')

    train_set = ImageInImageOutDataSet(root_folder=train_folder,
                                       data_shape=data_shape,
                                       not_roi_info=not_roi_info,
                                       processor=processor)

    valid_set = ImageInImageOutDataSet(root_folder=validation_folder,
                                       data_shape=data_shape,
                                       not_roi_info=not_roi_info,
                                       processor=processor)

    train_loader = DataLoader(train_set, batch_size=12, shuffle=True)

    valid_loader = DataLoader(valid_set, batch_size=12, shuffle=True)

    model = Unet(in_channels=1, out_channels=1)
    model = model.to(device)

    paras = dict(model.named_parameters())
    encoding_paras = []
    decoding_paras = []
    classified_paras = []
    for k, v in paras.items():
        if 'Encoding' in k:
            encoding_paras.append(k)
        elif 'Decoding' in k:
            decoding_paras.append(k)
        else:
            classified_paras.append(k)

    params = [
        {"params": encoding_paras, "lr": 0.001},
        {"params": decoding_paras, "lr": 0.005},
        {"params": classified_paras, "lr": 0.005}
    ]
    optimizer = torch.optim.Adam(params)
    train_loss = 0.0
    valid_loss = 0.0
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)

    for epoch in range(150):
        train_loss_list, valid_loss_list = [], []
        train_auc_list, valid_auc_list = [], []
        label_list, pred_list = [], []

        model.train()
        for i, train_data in enumerate(train_loader):
            inputs, labels = train_data
            input_0, input_1, input_2 = inputs
            roi, ece = labels
            image = torch.cat((input_0, input_1, input_2), dim=1)
            image, roi, ece = image.to(device), roi.to(device), ece.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # TODO:
            sigmoid_outputs = torch.sigmoid(outputs)
            label_list.extend(list(labels.cpu().numpy()[..., 0]))
            pred_list.extend(list(sigmoid_outputs.cpu().detach().numpy()[..., 0]))
            train_auc = get_auc(pred_list, label_list, draw=False)

            train_loss_list.append(loss.item())
            train_auc_list.append(train_auc)
            train_loss += loss.item()

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f, Train auc: %.4f' %
                      (epoch + 1, 150, i + 1, train_loss/10, sum(train_auc_list)/len(train_auc_list)))
                train_loss = 0.0

        model.eval()
        for i, valid_data in enumerate(valid_loader):
            inputs, labels = valid_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            valid_loss_list.append(loss.item())
            valid_loss += loss.item()

            # TODO:
            sigmoid_outputs = torch.sigmoid(outputs)
            label_list.extend(list(labels.cpu().numpy()[..., 0]))
            pred_list.extend(list(sigmoid_outputs.cpu().detach().numpy()[..., 0]))
            valid_auc = get_auc(pred_list, label_list, draw=False)
            valid_auc_list.append(valid_auc)
            valid_loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Valid Loss: %.4f, valid auc: %.4f' %
                      (epoch + 1, 150, i + 1, valid_loss/10, sum(valid_auc_list)/len(valid_auc_list)))
                valid_loss = 0.0

        # writer.add_scalars('Train_Val_Loss',
        #                    {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        # writer.add_scalars('Train_Val_Auc',
        #                    {'train_auc': np.mean(train_auc_list), 'val_auc': np.mean(valid_auc_list)}, epoch + 1)
        # writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list))

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(np.mean(valid_loss_list), model, save_path=model_save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # state = {
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict()
        # }
        # torch.save(state, PATH)
    # torch.save(model, model_save_path)

def Test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_shape = {'input_0': (1, 184, 184), 'output_0': (2,)}

    test_set = BinaryOneImageOneLabelTest(root_folder=test_folder, data_shape=data_shape)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # model = torch.load(model_path)
    model = Unet(in_channels=1, out_channels=1)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    label_list = []
    pred_list = []
    binary_list = []
    for i, test_data in enumerate(test_loader):
        inputs, labels = test_data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        binary_outputs = BinaryPred(outputs)
        label_list.append(np.squeeze(labels.cpu().numpy())[0])
        pred_list.append(np.squeeze(outputs.cpu().detach().numpy())[0])
        binary_list.append(np.squeeze(binary_outputs.cpu().detach().numpy())[0])

    auc = get_auc(pred_list, label_list)
    compute_confusion_matrix(binary_list, label_list, model_name='ECE-UNet')
    print(auc)


if __name__ == '__main__':
    # data_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0/Train'
    # case_list = os.listdir(data_path)
    # print(case_list)
    Train()
    # Test()
