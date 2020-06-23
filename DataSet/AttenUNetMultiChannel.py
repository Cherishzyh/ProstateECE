import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from T4T.Utility.Loader import ImageInImageOutDataSet
from T4T.Utility.ImageProcessor import ImageProcess2D
from MeDIT.DataAugmentor import random_2d_augment

from Metric.classification_statistics import get_auc
from Model.AttenUnet import AttenUNet
from Model.UNet import UNet

from DataSet.CheckPoint import EarlyStopping


train_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Train'
validation_folder = r'//home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Validation'
test_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Test'

model_save_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Model'
model_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Model/checkpoint.pt'

graph_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Model/logs/'
image_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0_multichannel/Model/image/'


def Train():
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)

    data_shape = {'input_0': (1, 184, 184), 'output_0': (2,)}

    not_roi_info = {'input_0': True, 'output_0': False}

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

    model = AttenUNet(in_channels=3, out_channels=1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0.0
    valid_loss = 0.0
    # criterion = nn.NLLLoss()
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_folder, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []

        model.train()
        for i, train_data in enumerate(train_loader):
            inputs, labels = train_data
            inputs, ece = inputs.to(device), labels.to(device)
            ece = torch.argmax(ece, dim=1)
            ece = ece.long()

            optimizer.zero_grad()

            class_out = model(inputs)


            loss = criterion(class_out, ece)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute auc
            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out.cpu().detach().numpy()[..., 0]))

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' % (epoch + 1, 150, i + 1, train_loss/10))
                train_loss = 0.0

        # train_auc = get_auc(class_pred_list, class_list, draw=False)
        # class_list, class_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, valid_data in enumerate(valid_loader):
                inputs, labels = valid_data
                inputs, ece = inputs.to(device), labels.to(device)
                ece = torch.argmax(ece, dim=1)
                ece = ece.long()

                classification_out = model(inputs)
                # class_out_log = torch.log(classification_out)

                loss = criterion(classification_out, ece)
                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                # compute auc
                class_list.extend(list(ece.cpu().numpy()))
                class_pred_list.extend(list(classification_out.cpu().detach().numpy()[..., 0]))

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d], Valid Loss: %.4f' % (epoch + 1, 150, i + 1, valid_loss/10))
                    valid_loss = 0.0
            # valid_auc = get_auc(class_pred_list, class_list, draw=False)

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        # writer.add_scalars('Train_Val_Auc',
        #                    {'train_auc': train_auc, 'val_auc': valid_auc}, epoch + 1)
        writer.close()
        #
        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list),)
              # 'Training Auc:', round(train_auc, 4),
              # 'Valid Auc:', round(valid_auc, 4))

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(np.mean(valid_loss_list), model, save_path=model_save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)

    data_shape = {'input_0': (1, 184, 184), 'output_0': (1, 184, 184)}

    not_roi_info = {'input_0': True, 'output_0': False}

    test_set = ImageInImageOutDataSet(root_folder=validation_folder,
                                       data_shape=data_shape,
                                       not_roi_info=not_roi_info,
                                       processor=processor)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    model = AttenUNet(in_channels=3, out_channels=1)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    ece_list = []
    class_out_list = []

    for i, test_data in enumerate(test_loader):
        inputs, labels = test_data
        inputs, ece = inputs.to(device), labels.to(device)

        class_out = model(inputs)

        ece_list.append(ece.cpu().detach().numpy()[0][0])
        class_out_list.append(class_out.cpu().detach().numpy()[0][0])

    auc = get_auc(class_out_list, ece_list)
    print(auc)



if __name__ == '__main__':
    # data_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0/Train'
    # case_list = os.listdir(data_path)
    # print(case_list)
    Train()
    # Test()