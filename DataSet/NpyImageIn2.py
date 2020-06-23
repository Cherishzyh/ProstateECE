import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from T4T.Utility.Data import *
from MeDIT.DataAugmentor import random_2d_augment

from Model.ResNetcbam import ResNet, Bottleneck
# from Model.ResNet50 import ResNet, Bottleneck
from DataSet.CheckPoint import EarlyStopping
from NPYFilePath import *
from Metric.classification_statistics import get_auc, draw_roc


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/Model/ResNet50cbam'
model_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/Model/ResNet50cbam/checkpoint.pt'
graph_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/Model/ResNet50cbam/logs'


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

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    ###########################################################
    validation_dataset.AddOne(Image2D(validation_t2_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_dwi_folder, shape=(184, 184)))
    validation_dataset.AddOne(Image2D(validation_adc_folder, shape=(184, 184)))

    validation_dataset.AddOne(Feature(csv_folder), is_input=False)
    validation_dataset.AddOne(Image2D(validation_roi_folder, shape=(184, 184), is_roi=True), is_input=False)
    validation_dataset.AddOne(Image2D(validation_prostate_folder, shape=(184, 184), is_roi=True), is_input=False)

    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

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
    train_loader, validation_loader = LoadTVData(random_2d_augment)
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0.0
    valid_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_path, comment='Net')

    for epoch in range(1000):
        train_loss_list, valid_loss_list = [], []
        class_list, class_pred_list = [], []

        model.train()
        for i, (inputs, outputs) in enumerate(train_loader):
            t2, dwi, adc, ece = inputs[0], inputs[1], inputs[2], np.squeeze(inputs[3], axis=1)
            inputs = torch.cat([t2, dwi, adc], axis=1)

            inputs = inputs.type(torch.FloatTensor).to(device)
            ece = ece.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()

            class_out = model(inputs)

            loss = criterion(class_out, ece)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute auc
            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out.cpu().detach().numpy()))

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' % (epoch + 1, 150, i + 1, train_loss / 10))
                train_loss = 0.0

        _, _, train_auc = get_auc(class_pred_list, class_list)
        class_list, class_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, (inputs, outputs) in enumerate(validation_loader):
                t2, dwi, adc, ece = inputs[0], inputs[1], inputs[2], np.squeeze(inputs[3], axis=1)
                inputs = torch.cat([t2, dwi, adc], axis=1)
                inputs = inputs.type(torch.FloatTensor).to(device)
                ece = ece.type(torch.FloatTensor).to(device)

                classification_out = model(inputs)

                loss = criterion(classification_out, ece)
                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                # compute auc
                class_list.extend(list(ece.cpu().numpy()))
                class_pred_list.extend(list(classification_out.cpu().detach().numpy()[..., 0]))

                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d], Valid Loss: %.4f' % (epoch + 1, 150, i + 1, valid_loss / 10))
                    valid_loss = 0.0
            _, _, valid_auc = get_auc(class_pred_list, class_list)

        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        writer.add_scalars('Train_Val_auc',
                           {'train_auc': train_auc,'val_auc': valid_auc}, epoch + 1)
        writer.close()

        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:',
              np.mean(valid_loss_list), 'Train auc:', train_auc, 'Valid auc:', valid_auc)

        scheduler.step(np.mean(valid_loss_list))
        early_stopping(valid_auc, model, save_path=model_folder, evaluation=max)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    import matplotlib.pyplot as plt
    test_loader = LoadTestData()
    train_loader, validation_loader = LoadTVData()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    fpr_list, tpr_list, auc_list = [], [], []
    name_list = ['train', 'validation', 'test']

    loader_list = [train_loader, validation_loader, test_loader]
    # loader_list = [test_loader]
    # with torch.no_grad():
    for loader in loader_list:
        class_list, class_pred_list = [], []
        for i, (inputs, outputs) in enumerate(loader):
            t2, dwi, adc, ece = inputs[0], inputs[1], inputs[2], np.squeeze(inputs[3], axis=1)
            inputs = torch.cat([t2, dwi, adc], axis=1)

            inputs = inputs.type(torch.FloatTensor).to(device)
            # numpy_data = inputs.cpu().detach().numpy()
            # for idx in range(numpy_data.shape[0]):
            #     data = np.squeeze(numpy_data[idx, 0, ...])
            #     plt.imshow(data, cmap='gray')
            #     plt.show()
            ece = ece.type(torch.FloatTensor).to(device)
            class_out = model(inputs)

            class_list.extend(list(ece.cpu().numpy()))
            class_pred_list.extend(list(class_out.cpu().detach().numpy()))
        fpr, tpr, auc = get_auc(class_pred_list, class_list)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)
    draw_roc(fpr_list, tpr_list, auc_list, name_list)


def FeatureMap():
    test_dataset = DataManager(random_2d_augment)
    test_dataset.AddOne(Image2D(t2_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(dwi_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(adc_folder, shape=(184, 184)))
    test_dataset.AddOne(Image2D(roi_folder, shape=(184, 184)))
    test_dataset.AddOne(Feature(csv_folder))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # with torch.no_grad():
    for i, (inputs, outputs) in enumerate(test_loader):
        t2, dwi, adc, roi, ece = inputs[0], inputs[1], inputs[2], inputs[3], np.squeeze(inputs[3], axis=1)
        np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/t2.npy',
                t2)
        np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/dwi.npy',
                dwi)
        np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/adc.npy',
                adc)
        np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/roi.npy',
                roi)

        inputs = torch.cat([t2, dwi, adc], axis=1)
        inputs = inputs.type(torch.FloatTensor).to(device)
        class_out = model(inputs)


def ShowPicture():
    from Metric.Dice import Dice
    import matplotlib.pyplot as plt
    dice = Dice()
    train_loader, validation_loader = LoadTVData(is_test=True)
    test_loader = LoadTestData(is_test=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model = MultiTaskModel(in_channels=3, out_channels=1).to(device)
    model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
    model.load_state_dict(torch.load(model_path))
    ece_pre_list = []
    ece_list = []
    for i, (inputs, outputs) in enumerate(train_loader):
        t2, dwi, adc = inputs[0], inputs[1], inputs[2],
        ece, roi, prostate = np.squeeze(outputs[0], axis=1), outputs[1].to(device), outputs[2].to(device)

        inputs = torch.cat([t2, dwi, adc], axis=1)
        inputs = inputs.type(torch.FloatTensor).to(device)

        ece = ece.type(torch.FloatTensor).to(device)

        # roi_out, prostate_out, class_out = model(inputs)
        class_out = model(inputs)
        ece_pre_list.append(class_out.sigmoid().cpu().detach().numpy()[0][0])
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
    plt.hist(ece_pre_list)
    plt.show()


if __name__ == '__main__':
    # Train()
    # Test()
    # FeatureMap()
    ShowPicture()