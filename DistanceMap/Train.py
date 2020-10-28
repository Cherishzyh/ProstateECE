import os
import cv2
import numpy as np
import shutil
import matplotlib.pyplot as plt


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

from MyModel.UNet import UNet
from DataSet.MyDataLoader import LoadTVData, LoadTestData
from Metric.Loss import BCEFocalLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY'
model_folder = r'/home/zhangyihong/Documents/ProstateECE/Model/DistanceMap/DistanceMapSegPre'
model_path = r'/home/zhangyihong/Documents/ProstateECE/Model/DistanceMap/DistanceMapSegPre/checkpoint.pt'
graph_path = r'/home/zhangyihong/Documents/ProstateECE/Model/DistanceMap/DistanceMapSegPre/logs'


def LoadTVData(folder, is_test=False, setname=None):
    if setname is None:
        setname = ['Train', 'Validation']

    roi_folder = os.path.join(folder, 'RoiSlice')
    prostate_folder = os.path.join(folder, 'ProstateSlice')
    distance_folder = os.path.join(folder, 'DistanceMap')

    train_pca_folder = os.path.join(roi_folder, setname[0])
    train_prostate_folder = os.path.join(prostate_folder, setname[0])
    train_distance_folder = os.path.join(distance_folder, setname[0])

    validation_pca_folder = os.path.join(roi_folder, setname[1])
    validation_prostate_folder = os.path.join(prostate_folder, setname[1])
    validation_distance_folder = os.path.join(distance_folder, setname[1])

    if is_test:
        train_dataset = DataManager()
        validation_dataset = DataManager()
    else:
        train_dataset = DataManager(config_example)
        validation_dataset = DataManager(config_example)

    ###########################################################
    train_dataset.AddOne(Image2D(train_pca_folder, shape=(184, 184), is_roi=True))
    train_dataset.AddOne(Image2D(train_prostate_folder, shape=(184, 184), is_roi=True))

    train_dataset.AddOne(Image2D(train_distance_folder, shape=(184, 184), is_roi=True), is_input=False)

    ###########################################################

    validation_dataset.AddOne(Image2D(validation_pca_folder, shape=(184, 184), is_roi=True))
    validation_dataset.AddOne(Image2D(validation_prostate_folder, shape=(184, 184), is_roi=True))

    validation_dataset.AddOne(Image2D(validation_distance_folder, shape=(184, 184), is_roi=True), is_input=False)

    ###########################################################
    if is_test:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=12, shuffle=True)

    return train_loader, validation_loader


def LoadTestData(folder):
    roi_folder = os.path.join(folder, 'RoiSlice')
    prostate_folder = os.path.join(folder, 'ProstateSlice')
    distance_folder = os.path.join(folder, 'DistanceMap')

    test_pca_folder = os.path.join(roi_folder, 'Test')
    test_prostate_folder = os.path.join(prostate_folder, 'Test')
    test_distance_folder = os.path.join(distance_folder, 'Test')

    test_dataset = DataManager()

    ###########################################################
    test_dataset.AddOne(Image2D(test_pca_folder, shape=(184, 184), is_roi=True))
    test_dataset.AddOne(Image2D(test_prostate_folder, shape=(184, 184), is_roi=True))

    test_dataset.AddOne(Image2D(test_distance_folder, shape=(184, 184), is_roi=True), is_input=False)

    ###########################################################

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_loader


def ClearGraphPath():
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def Train():
    # ClearGraphPath()
    train_loader, validation_loader = LoadTVData(folder=data_folder, is_test=False)
    model = UNet(in_channels=2, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=50, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        train_loss, valid_loss = 0.0, 0.0

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            pca, prostate = inputs[0], inputs[1]
            distance_map = outputs.to(device)

            inputs = torch.cat([pca, prostate], dim=1)
            inputs = inputs.float().to(device)

            distance_out = model(inputs)

            loss = criterion(distance_out, distance_map)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d],  Loss: %.4f' %
                      (epoch + 1, 1000, i + 1, train_loss / 10))
                train_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                pca, prostate = inputs[0], inputs[1]
                distance_map = outputs.to(device)

                inputs = torch.cat([pca, prostate], dim=1)
                inputs = inputs.float().to(device)

                distance_out = model(inputs)

                loss = criterion(distance_out, distance_map)

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d],  Loss: %.4f' %
                          (epoch + 1, 1000, i + 1, valid_loss / 10))
                    valid_loss = 0.0

        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list))

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(sum(valid_loss_list)/len(valid_loss_list), model, save_path=model_folder, evaluation=min)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    from Metric.Dice import Dice
    import matplotlib.pyplot as plt
    test_loader = LoadTestData(data_folder)
    train_loader, validation_loader = LoadTVData(data_folder, is_test=True)

    model = UNet(in_channels=2, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [train_loader, validation_loader, test_loader]

    # with torch.no_grad():
    model.eval()
    # for name_num, loader in enumerate(loader_list):
    distance_map_list, distance_out_list = [], []
    prostate_dice, roi_dice = [], []
    for i, (inputs, outputs) in enumerate(test_loader):
        pca, prostate = inputs[0], inputs[1]
        distance_map = outputs.to(device)

        inputs = torch.cat([pca, prostate], dim=1)
        inputs = inputs.float().to(device)

        distance_out = model(inputs)

        distance_out_np = distance_out.cpu().detach().numpy()
        distance_map_np = distance_map.cpu().numpy()
        distance_out_list.extend(list(distance_out_np))
        distance_map_list.extend(list(distance_map_np))

        plt.subplot(121)
        plt.imshow(np.squeeze(distance_map_np), cmap='jet', vmin=0., vmax=1.)
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(np.squeeze(distance_out_np), cmap='jet', vmin=0., vmax=1.)
        plt.colorbar()
        plt.show()




        # for idx in range(len(roi_list)):
        #     prostate_dice.append(dice(prostate_list[idx], prostate_pred_list[idx]).numpy())
        # print('average dice of roi in', name_list[name_num], ':', sum(roi_dice)/len(roi_dice))
        # print('average dice of prostate in', name_list[name_num], ':', sum(prostate_dice) / len(prostate_dice))
        # plt.hist(roi_dice)
        # plt.title('Dice of Prostate Cancer in ' + name_list[name_num])
        # plt.show()
        #
        # plt.hist(prostate_dice)
        # plt.title('Dice of Prostate in ' + name_list[name_num])
        # plt.show()





if __name__ == '__main__':
    # Train()
    Test()
    # ShowPicture()
    # FeatureMap()