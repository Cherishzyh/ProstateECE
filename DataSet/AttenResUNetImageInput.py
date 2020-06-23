import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from T4T.Utility.Loader import ImageInImageOutDataSet
from T4T.Utility.ImageProcessor import ImageProcess2D
from MeDIT.DataAugmentor import random_2d_augment

from Metric.Dice import Dice
from Metric.classification_statistics import get_auc, compute_confusion_matrix
from Model.AttenUNetMultiTask import AttenUNetMultiTask2D
from Model.AttenUnet import AttenUNet
# from FilePath import *
from DataSet.CheckPoint import EarlyStopping


train_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Train'
validation_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Validation'
test_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Test'

model_save_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Model'
model_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Model/checkpoint.pt'

graph_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Model/logs/'
image_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_1_3D/Model/image/'
def BinaryPred(prediction):
    one = torch.ones_like(prediction)
    zero = torch.zeros_like(prediction)
    binary_prediction = torch.where(prediction > 0.5, one, zero)
    return binary_prediction


def Train():
    device = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')

    processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)

    data_shape = {'input_0': (3, 184, 184), 'output_0': (1, 184, 184), 'output_1': (2,)}

    not_roi_info = {'input_0': True, 'output_0': False, 'output_1': False}

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

    model = AttenUNetMultiTask2D(in_channels=3, out_channels=1)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = 0.0
    valid_loss = 0.0
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.NLLLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=100, verbose=True)
    writer = SummaryWriter(log_dir=graph_folder, comment='Net')
    dice = Dice()

    for epoch in range(150):
        train_loss_list, valid_loss_list = [], []
        train_dice_list, valid_dice_list = [], []
        class_list, class_pred_list = [], []
        seg_list, seg_pred_list = [], []

        model.train()
        for i, train_data in enumerate(train_loader):
            inputs, labels = train_data
            roi, ece = labels
            inputs, roi, ece = inputs.to(device), roi.to(device), ece.to(device)
            ece = torch.argmax(ece, dim=1)
            ece = ece.long()

            optimizer.zero_grad()

            seg_out, class_out = model(inputs)

            loss1 = criterion1(seg_out, roi)
            loss2 = criterion2(class_out, ece)
            loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_list.append(loss.item())

            # compute dice
            seg_out = BinaryPred(seg_out)
            seg_list.extend(list(torch.squeeze(roi)))
            seg_pred_list.extend(list(torch.squeeze(seg_out)))

            # compute auc
            # class_list.extend(list(ece.cpu().numpy()[..., 0]))
            # class_pred_list.extend(list(class_out.cpu().detach().numpy()[..., 0]))

            if (i + 1) % 10 == 0:
                print('Epoch [%d / %d], Iter [%d], Train Loss: %.4f' %
                      (epoch + 1, 150, i + 1, train_loss/10))
                train_loss = 0.0

        # train_auc = get_auc(class_pred_list, class_list, draw=False)
        for index in range(len(seg_list)):
            train_dice_list.append(dice.forward(seg_pred_list[index], seg_list[index]))

        # class_list, class_pred_list = [], []
        # seg_list, seg_pred_list = [], []

        model.eval()
        with torch.no_grad():
            for i, valid_data in enumerate(valid_loader):
                inputs, labels = valid_data
                roi, ece = labels
                inputs, roi, ece = inputs.to(device), roi.to(device), ece.to(device)
                ece = torch.argmax(ece, dim=1)
                ece = ece.long()

                segmentation_out, classification_out = model(inputs)

                loss1 = criterion1(segmentation_out, roi)
                loss2 = criterion2(classification_out, ece)
                loss = loss1 + loss2
                valid_loss += loss.item()
                valid_loss_list.append(loss.item())

                seg_out = BinaryPred(segmentation_out)
                seg_list.extend(list(torch.squeeze(seg_out)))
                seg_pred_list.extend(list(torch.squeeze(roi)))

                # compute auc
                # class_list.extend(list(ece.cpu().numpy()[..., 0]))
                # class_pred_list.extend(list(classification_out.cpu().detach().numpy()[..., 0]))


                if (i + 1) % 10 == 0:
                    print('Epoch [%d / %d], Iter [%d], Valid Loss: %.4f' %
                          (epoch + 1, 150, i + 1, valid_loss/10))
                    valid_loss = 0.0

            # valid_auc = get_auc(class_pred_list, class_list, draw=False)
            for idx in range(len(seg_list)):
                valid_dice_list.append(dice.forward(seg_pred_list[idx], seg_list[idx]))


        writer.add_scalars('Train_Val_Loss',
                           {'train_loss': np.mean(train_loss_list), 'val_loss': np.mean(valid_loss_list)}, epoch + 1)
        # writer.add_scalars('Train_Val_Auc',
        #                    {'train_auc': train_auc, 'val_auc': valid_auc}, epoch + 1)
        writer.add_scalars('Train_Val_Dice',
                           {'train_dice': torch.mean(torch.stack(train_dice_list)), 'val_dice': torch.mean(torch.stack(valid_dice_list))}, epoch + 1)
        writer.close()
        #
        print('Epoch:', epoch + 1, 'Training Loss:', np.mean(train_loss_list), 'Valid Loss:', np.mean(valid_loss_list))
        # print('Training Auc:', round(train_auc, 4), 'Valid Auc:', round(valid_auc, 4))
        print('Training Dice:', torch.mean(torch.stack(train_dice_list)).cpu().numpy(), 'Valid Dice:', torch.mean(torch.stack(valid_dice_list)).cpu().numpy())
        scheduler.step(np.mean(valid_loss_list))
        early_stopping(np.mean(valid_loss_list), model, save_path=model_save_path)

        if early_stopping.early_stop:
            print("Early stopping")
            break


def Test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = ImageProcess2D(reverse_channel=False, augment_param=random_2d_augment)

    data_shape = {'input_0': (1, 184, 184), 'output_0': (1, 184, 184), 'output_1': (2,)}

    not_roi_info = {'input_0': True, 'output_0': False, 'output_1': False}

    test_set = ImageInImageOutDataSet(root_folder=test_folder,
                                       data_shape=data_shape,
                                       not_roi_info=not_roi_info,
                                       processor=processor)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

    # model = torch.load(model_path)
    model = AttenUNetMultiTask2D(in_channels=3, out_channels=1)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    roi_list = []
    ece_list = []
    class_out_list = []
    binary_seg_list = []
    test_dice = []
    dice = Dice()

    for i, test_data in enumerate(test_loader):
        inputs, labels = test_data
        roi, ece = labels
        inputs, roi, ece = inputs.to(device), roi.to(device), ece.to(device)

        seg_out, class_out = model(inputs)

        binary_outputs = BinaryPred(seg_out)
        roi_list.append(np.squeeze(roi.cpu().numpy()))
        ece_list.append(ece.cpu().numpy()[0][0])

        binary_seg_list.append(np.squeeze(binary_outputs.cpu().detach().numpy()))
        class_out_list.append(class_out.cpu().detach().numpy()[0][0])

        # plt.subplot(121)
        # plt.imshow(torch.squeeze(inputs).cpu().detach().numpy(), cmap='gray')
        # plt.contour(torch.squeeze(roi).cpu().detach().numpy(), colors='r')
        # plt.subplot(122)
        # plt.imshow(torch.squeeze(binary_outputs).cpu().detach().numpy(), cmap='gray')
        # plt.savefig(os.path.join(image_folder, str(i) + '.jpg'))
        # plt.close()
        test_dice.append((dice.forward(np.squeeze(roi), np.squeeze(binary_outputs))).cpu().detach().numpy())

    auc = get_auc(class_out_list, ece_list)
    print(auc)
    print(sum(test_dice) / len(test_dice))


if __name__ == '__main__':
    # data_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/input_0_output_0/Train'
    # case_list = os.listdir(data_path)
    # print(case_list)
    # Train()
    Test()
