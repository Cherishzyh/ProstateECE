import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from T4T.Utility.Data import *
from MeDIT.DataAugmentor import random_2d_augment
from T4T.Utility.Loss import DiceLoss
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01

from Model.MultiTaskModel2 import MultiTaskModel
from DataSet.CheckPoint import EarlyStopping

from NPYFilePath import *
from Metric.classification_statistics import get_auc, draw_roc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def BinaryPred(prediction):
    one = torch.ones_like(prediction)
    zero = torch.zeros_like(prediction)
    binary_prediction = torch.where(prediction > 0.5, one, zero)
    return binary_prediction


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

    train_dataset.AddOne(Feature(csv_folder), is_input=False)
    train_dataset.AddOne(Image2D(train_roi_folder, shape=(184, 184), is_roi=True), is_input=False)
    train_dataset.AddOne(Image2D(train_prostate_folder, shape=(184, 184), is_roi=True), is_input=False)

    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)

    ###########################################################
    validation_dataset.AddOne(Image2D(validation_t2_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_dwi_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_adc_folder, shape=(184, 184)))

    validation_dataset.AddOne(Feature(csv_folder), is_input=False)
    validation_dataset.AddOne(Image2D(validation_roi_folder, shape=(184, 184), is_roi=True), is_input=False)
    validation_dataset.AddOne(Image2D(validation_prostate_folder, shape=(184, 184), is_roi=True), is_input=False)

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

    test_dataset.AddOne(Feature(csv_folder), is_input=False)
    test_dataset.AddOne(Image2D(test_roi_folder, shape=(184, 184), is_roi=True), is_input=False)
    test_dataset.AddOne(Image2D(test_prostate_folder, shape=(184, 184), is_roi=True), is_input=False)


    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    return test_loader


def Train():
    train_loader, validation_loader = LoadTVData()
    model = MultiTaskModel(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss1 = 0.0
    train_loss2 = 0.0
    train_loss3 = 0.0
    train_loss = 0.0
    valid_loss1 = 0.0
    valid_loss2 = 0.0
    valid_loss3 = 0.0
    valid_loss = 0.0
    seg_criterion1 = DiceLoss()
    seg_criterion2 = DiceLoss()
    cla_criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss1_list, valid_loss1_list = [], []
        train_loss2_list, valid_loss2_list = [], []
        train_loss3_list, valid_loss3_list = [], []
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc = inputs[0], inputs[1], inputs[2],
            ece, roi, prostate = np.squeeze(outputs[0], axis=1), outputs[1].to(device), outputs[2].to(device)

            inputs = torch.cat([t2, dwi, adc], axis=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            roi_out, prostate_out, class_out = model(inputs)

            loss1 = seg_criterion1(roi_out, roi)
            loss2 = seg_criterion2(prostate_out, prostate)
            loss3 = cla_criterion(class_out, ece)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            train_loss1 += loss1.item()
            train_loss1_list.append(loss1.item())

            train_loss2 += loss2.item()
            train_loss2_list.append(loss2.item())

            train_loss3 += loss3.item()
            train_loss3_list.append(loss3.item())

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute auc
            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out.cpu().detach().numpy()))

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Cancer train Loss: %.4f, Prostate train Loss: %.4f, ECE train Loss: %.4f, Loss: %.4f' %
                      (epoch + 1, 1000, i + 1, train_loss1 / 10, train_loss2 / 10, train_loss3 / 10, train_loss / 10))
                train_loss = 0.0
                train_loss1 = 0.0
                train_loss2 = 0.0
                train_loss3 = 0.0

        _, _, train_auc = get_auc(class_pred_list, class_list)
        class_list, class_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc = inputs[0], inputs[1], inputs[2],
                ece, roi, prostate = np.squeeze(outputs[0], axis=1), outputs[1].to(device), outputs[2].to(device)

                inputs = torch.cat([t2, dwi, adc], axis=1)
                inputs = inputs.type(torch.FloatTensor).to(device)

                ece = ece.type(torch.FloatTensor).to(device)

                roi_out, prostate_out, class_out = model(inputs)

                loss1 = seg_criterion1(roi_out, roi)
                loss2 = seg_criterion2(prostate_out, prostate)
                loss3 = cla_criterion(class_out, ece)
                loss = loss1 + loss2 + loss3

                valid_loss1 += loss1.item()
                valid_loss1_list.append(loss1.item())

                valid_loss2 += loss2.item()
                valid_loss2_list.append(loss2.item())

                valid_loss3 += loss3.item()
                valid_loss3_list.append(loss3.item())

                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                # compute auc
                class_list.extend(list(ece.cpu().numpy()))
                class_pred_list.extend(list(class_out.cpu().detach().numpy()[..., 0]))

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d], Cancer validation Loss: %.4f, Prostate validation Loss: %.4f, ECE validation Loss: %.4f, Loss: %.4f' %
                          (epoch + 1, 1000, i + 1, valid_loss1 / 10, valid_loss2 / 10, valid_loss3 / 10, valid_loss / 10))
                    valid_loss1 = 0.0
                    valid_loss2 = 0.0
                    valid_loss3 = 0.0
                    valid_loss = 0.0
            _, _, valid_auc = get_auc(class_pred_list, class_list)

        for index, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name+'_grad', param.grad.cpu().data.numpy(), epoch+1)
                writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

        writer.add_scalars('Train_Val_Loss1',
                           {'train_cancer_dice_loss': np.mean(train_loss1_list), 'val_cancer_dice_loss': np.mean(valid_loss1_list)}, epoch + 1)
        writer.add_scalars('Train_Val_Loss2',
                           {'train_prostate_dice_loss': np.mean(train_loss2_list), 'val_prostate_dice_loss': np.mean(valid_loss2_list)}, epoch + 1)
        writer.add_scalars('Train_Val_Loss3',
                           {'train_bce_loss': np.mean(train_loss3_list), 'val_bce_loss': np.mean(valid_loss3_list)}, epoch + 1)
        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.add_scalars('Train_Val_auc',
                           {'train_auc': train_auc, 'val_auc': valid_auc}, epoch + 1)
        writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:',
              np.mean(valid_loss_list), 'Train auc:', train_auc, 'Valid auc:', valid_auc)

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(valid_auc, model, save_path=model_folder, evaluation=max)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    from Metric.Dice import Dice
    import matplotlib.pyplot as plt
    dice = Dice()
    test_loader = LoadTestData(is_test=True)
    train_loader, validation_loader = LoadTVData(is_test=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))

    fpr_list, tpr_list, auc_list = [], [], []

    name_list = ['Train', 'Validation', 'Test']
    loader_list = [train_loader, validation_loader, test_loader]

    # with torch.no_grad():
    for name_num, loader in enumerate(loader_list):
        class_list, class_pred_list = [], []
        prostate_list, prostate_pred_list = [], []
        roi_list, roi_pred_list = [], []
        prostate_dice, roi_dice = [], []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc = inputs[0], inputs[1], inputs[2],
            ece, roi, prostate = np.squeeze(outputs[0], axis=1), outputs[1].to(device), outputs[2].to(device)

            inputs = torch.cat([t2, dwi, adc], axis=1)
            inputs = inputs.type(torch.FloatTensor).to(device)

            ece = ece.type(torch.FloatTensor).to(device)

            roi_out, prostate_out, class_out = model(inputs)

            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out.cpu().detach().numpy()))

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
        print('average dice of roi:', sum(roi_dice)/len(roi_dice))
        print('average dice of prostate:', sum(prostate_dice) / len(prostate_dice))
        plt.hist(roi_dice)
        plt.title('Dice of Prostate Cancer in ' + name_list[name_num])
        plt.show()

        plt.hist(prostate_dice)
        plt.title('Dice of Prostate in ' + name_list[name_num])
        plt.show()
        fpr, tpr, auc = get_auc(class_pred_list, class_list)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    draw_roc(fpr_list, tpr_list, auc_list, name_list)


def ShowPicture():
    from Metric.Dice import Dice
    import matplotlib.pyplot as plt
    dice = Dice()
    train_loader, validation_loader = LoadTVData(is_test=True)
    test_loader = LoadTestData(is_test=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    ece_pre_list = []
    ece_list = []
    for i, (inputs, outputs) in enumerate(test_loader):
        t2, dwi, adc = inputs[0], inputs[1], inputs[2],
        ece, roi, prostate = np.squeeze(outputs[0], axis=1), outputs[1].to(device), outputs[2].to(device)

        inputs = torch.cat([t2, dwi, adc], axis=1)
        inputs = inputs.type(torch.FloatTensor).to(device)

        ece = ece.type(torch.FloatTensor).to(device)

        roi_out, prostate_out, class_out = model(inputs)
        ece_pre_list.append(class_out.cpu().detach().numpy()[0][0])
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



def viz(module, input):
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i])
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MultiTaskModel(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)

    adc = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\AdcSlice\Test\BHX^bao han xiu ^^6875-5_slice9.npy')
    t2 = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\T2Slice\Test\BHX^bao han xiu ^^6875-5_slice9.npy')
    dwi = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\DwiSlice\Test\BHX^bao han xiu ^^6875-5_slice9.npy')
    # roi = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\RoiSlice\Test\BHX^bao han xiu ^^6875-5_slice9.npy')
    # prostate = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\ProstateSlice\Test\BHX^bao han xiu ^^6875-5_slice9.npy')
    ece = 0.0
    inputs = torch.cat([t2, dwi, adc], axis=1)
    inputs = inputs.type(torch.FloatTensor).to(device)
    ece = ece.type(torch.FloatTensor).to(device)
    with torch.no_grad():
        model(inputs)



if __name__ == '__main__':
    # Train()
    Test()
    # ShowPicture()
    # main()