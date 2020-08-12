import os
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

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

from MyModel.ResNet50 import ResNet, Bottleneck
from NPYFilePath import *
from Metric.classification_statistics import get_auc, draw_roc


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model_folder = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNet505InputOneHot/'
model_path = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNet505InputOneHot/checkpoint.pt'
graph_path = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNet505InputOneHot/logs'


def ClearGraphPath():
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def LoadTVData(is_test=False):
    if is_test:
        train_dataset = DataManager()
        validation_dataset = DataManager()
    else:
        train_dataset = DataManager(random_2d_augment)
        validation_dataset = DataManager(random_2d_augment)

    ###########################################################
    train_dataset.AddOne(Image2D(train_t2_folder, shape=(184, 184)))
    train_dataset.AddOne(Image2D(train_dwi_folder, shape=(184, 184)))
    train_dataset.AddOne(Image2D(train_adc_folder, shape=(184, 184)))
    train_dataset.AddOne(Image2D(train_roi_folder, shape=(184, 184), is_roi=True))
    train_dataset.AddOne(Image2D(train_prostate_folder, shape=(184, 184), is_roi=True))

    train_dataset.AddOne(Feature(csv_folder), is_input=False)

    ###########################################################
    validation_dataset.AddOne(Image2D(validation_t2_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_dwi_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_adc_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_roi_folder, shape=(184, 184), is_roi=True))
    validation_dataset.AddOne(Image2D(validation_prostate_folder, shape=(184, 184), is_roi=True))

    validation_dataset.AddOne(Feature(csv_folder), is_input=False)

    ###########################################################
    if is_test:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=12, shuffle=True)

    return train_loader, validation_loader


def LoadTestData(is_test=False):
    if is_test:
        test_dataset = DataManager()
    else:
        test_dataset = DataManager(random_2d_augment)
    test_dataset.AddOne(Image2D(test_t2_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(test_dwi_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(test_adc_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(test_roi_folder, shape=(184, 184), is_roi=True))
    test_dataset.AddOne(Image2D(test_prostate_folder, shape=(184, 184), is_roi=True))

    test_dataset.AddOne(Feature(csv_folder), is_input=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_loader


def Train():
    ClearGraphPath()
    train_loader, validation_loader = LoadTVData()
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0.0
    valid_loss = 0.0
    cla_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            ece = np.argmax(outputs, axis=1)

            inputs = torch.cat([t2, dwi, adc, roi, prostate], dim=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            class_out, _ = model(inputs)
            # class_out = torch.squeeze(class_out, dim=1)
            class_out_softmax = nn.functional.softmax(class_out, dim=1)

            loss = cla_criterion(class_out, ece)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute auc
            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out_softmax.cpu().detach().numpy()[..., 1]))

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' %(epoch + 1, 1000, i + 1, train_loss / 10))
                train_loss = 0.0

        _, _, train_auc = get_auc(class_pred_list, class_list)
        class_list, class_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
                ece = np.argmax(outputs, axis=1)

                inputs = torch.cat([t2, dwi, adc, roi, prostate], axis=1)
                inputs = inputs.type(torch.FloatTensor).to(device)

                ece = ece.type(torch.LongTensor).to(device)

                class_out, _ = model(inputs)
                # class_out = torch.squeeze(class_out, dim=1)
                class_out_softmax = nn.functional.softmax(class_out, dim=1)

                loss = cla_criterion(class_out, ece)

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                # compute auc
                class_list.extend(list(ece.cpu().numpy()))
                class_pred_list.extend(list(class_out_softmax.cpu().detach().numpy()[..., 1]))

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d],  Valid Loss: %.4f' %(epoch + 1, 1000, i + 1, valid_loss / 10))

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
        # early_stopping(sum(valid_loss_list)/len(valid_loss_list), model, save_path=model_folder, evaluation=min)
        early_stopping(valid_auc, model, save_path=model_folder, evaluation=max)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    test_loader = LoadTestData(is_test=True)
    train_loader, validation_loader = LoadTVData(is_test=True)

    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))

    fpr_list, tpr_list, auc_list = [], [], []

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [train_loader, validation_loader, test_loader]

    model.eval()
    # with torch.no_grad():
    for name_num, loader in enumerate(loader_list):
        class_list, class_pred_list = [], []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            ece = np.argmax(outputs, axis=1)

            inputs = torch.cat([t2, dwi, adc, roi, prostate], axis=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.LongTensor).to(device)

            class_out = model(inputs)
            class_out = torch.squeeze(class_out, dim=1)
            class_out_softmax = nn.functional.softmax(class_out, dim=1)

            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out_softmax.cpu().detach().numpy()[..., 1]))

        fpr, tpr, auc = get_auc(class_pred_list, class_list)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    draw_roc(fpr_list, tpr_list, auc_list, name_list)


def ShowPicture():
    from Metric.Dice import Dice

    dice = Dice()
    train_loader, validation_loader = LoadTVData(is_test=True)
    test_loader = LoadTestData(is_test=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))
    ece_pre_list = []
    ece_list = []
    for i, (inputs, outputs) in enumerate(test_loader):
        t2, dwi, adc, roi, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        ece = np.squeeze(outputs, axis=1)

        inputs = torch.cat([t2, dwi, adc, roi, prostate], axis=1)
        inputs = inputs.type(torch.FloatTensor).to(device)

        ece = ece.type(torch.FloatTensor).to(device)

        class_out, _ = model(inputs)
        class_out_softmax = nn.functional.softmax(class_out, dim=1)
        ece_pre_list.append(class_out_softmax.cpu().detach().numpy()[0][0])
        ece_list.append(ece.cpu().numpy()[0][0])

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
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))

    adc = np.load(os.path.join(test_adc_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    dwi = np.load(os.path.join(test_dwi_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    t2 = np.load(os.path.join(test_t2_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    prostate = np.load(os.path.join(test_prostate_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))
    cancer = np.load(os.path.join(test_roi_folder, 'BHX^bao han xiu ^^6875-5_slice9.npy'))

    inputs = np.concatenate([t2, dwi, adc], axis=0)
    inputs = inputs[np.newaxis, :]
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor).to(device)

    x, feature_map = model(inputs)

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