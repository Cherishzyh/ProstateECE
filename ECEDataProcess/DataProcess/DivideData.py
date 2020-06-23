import os
import random
import shutil
import numpy as np

from MeDIT.SaveAndLoad import LoadH5

from FilePath import *
from ECEDataProcess.DataProcess.SelectDWI import NearTrueB
# 正负样本比例1:3
# b值800的在test

def GetBValue(case, b_value):
    if case == 'WYB^wu yi bao' or case == 'ZYB^zhang yun bao':
        return 0
    case_path = os.path.join(process_folder, case)
    if os.path.exists(os.path.join(case_path, 'dwi.nii')):
        bval_path = os.path.join(case_path, 'dwi.bval')
        bval = open(bval_path, 'r')
        b_list = bval.read().split()
        if len(b_list) == 1:
            return b_list[0]
        else:
            b, index = NearTrueB(b_list)
            return b

    elif os.path.exists(os.path.join(case_path, 'dki.nii')):
        bval_path = os.path.join(case_path, 'dki.bval')
        bval = open(bval_path, 'r')
        b_list = bval.read().split()
        if len(b_list) == 1:
            return b_list[0]
        else:
            b, index = NearTrueB(b_list)
            return b

    else:
        for b in reversed(b_value):
            if os.path.exists(os.path.join(case_path, 'dwi_b' + str(b) + '.nii')):
                return b

def StatisticsECE(folder):
    ece_number = 0
    no_ece_number = 0

    case_list = os.listdir(folder)
    for case in case_list:
        case_path = os.path.join(folder, case)
        if not os.path.isdir(case_path):
            ece = LoadH5(case_path, tag=['output_1'], data_type=np.uint8)
            if all(ece == np.array([0, 1])):
                no_ece_number += 1
            elif all(ece == np.array([1, 0])):
                ece_number += 1
            else:
                print(case)
    return ece_number, no_ece_number

def DivideH5Data(folder, des_folder):
    case_list = os.listdir(folder)
    random.shuffle(case_list)
    ece_number = 0
    no_ece_number = 0
    for case in case_list:
        case_path = os.path.join(folder, case)
        des_path = os.path.join(des_folder, case)
        if not os.path.isdir(case_path):
            case_name = case[:case.index('.npy')]
            ece = info.loc[case_name, 'pECE']

            if all(ece == np.array([1])):
                if no_ece_number == 10:
                    continue
                else:
                    no_ece_number += 1
                    shutil.move(case_path, des_path)
            elif all(ece == np.array([0])):
                if ece_number == 30:
                    continue
                else:
                    ece_number += 1
                    shutil.move(case_path, des_path)
            else:
                print(case)

def DivideInput0Data(folder, des_folder, dataset=''):
    folder = os.path.join(folder, dataset)
    des_folder_dataset = os.path.join(des_folder, dataset)

    case_list = os.listdir(folder)

    for case in case_list:
        case_path = os.path.join(des_folder, case)
        des_path = os.path.join(des_folder_dataset, case)

        if not os.path.isdir(case_path):
            shutil.move(case_path, des_path)
        else:
            print(case)


def ComputeV(folder):
    case_list = os.listdir(folder)
    volume_list = []
    for case in case_list:
        case_path = os.path.join(folder, case)
        label = LoadH5(case_path, tag=['output_0'], data_type=np.uint8)
        volume_list.append(np.sum(label))
    return sum(volume_list) / len(volume_list)


def Divide(des_folder, folder):
    des_train_folder = os.path.join(des_folder, 'Train')
    train_folder = os.path.join(folder, 'Train')

    des_validation_folder = os.path.join(des_folder, 'validation')
    validation_folder = os.path.join(folder, 'validation')

    des_test_folder = os.path.join(des_folder, 'Test')
    test_folder = os.path.join(folder, 'Test')

    case_list = os.listdir(des_folder)

    train_case_list = os.listdir(train_folder)
    validation_case_list = os.listdir(validation_folder)
    test_case_list = os.listdir(test_folder )

    for case in case_list:
        if case in train_case_list:
            shutil.move(os.path.join(des_folder, case), os.path.join(des_train_folder, case))
        elif case in validation_case_list:
            shutil.move(os.path.join(des_folder, case), os.path.join(des_validation_folder, case))
        elif case in test_case_list:
            shutil.move(os.path.join(des_folder, case), os.path.join(des_test_folder, case))
        else:
            print(case)


def CopyData(des_folder, folder):
    case_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/ProstateSlice/PreValid'
    case_list = os.listdir(case_folder)
    for case in case_list:
        case_path = os.path.join(folder, case)
        des_case_path = os.path.join(des_folder, case)
        shutil.move(case_path, des_case_path)


if __name__ == '__main__':
    import pandas as pd
    # b_value = ['0', '50', '700', '750', '1400', '1500']
    # case_list = os.listdir(process_folder)
    # for case in case_list:
    #     b = GetBValue(case, b_value)
    #     if isinstance(b, str):
    #         if float(b) < 1200:
    #             print(case)

    # csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/csv/ece.csv'
    cnn_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/RoiSlice/Test'
    pre_train_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/RoiSlice/PreValid'
    #
    # info = pd.read_csv(csv_path, usecols=['case', 'pECE'], index_col=['case'])
    #
    # DivideH5Data(cnn_folder, pre_train_folder)
    # DivideInput0Data(cnn_folder, input_0_output_0_path, dataset='Test')

    # ece_number, no_ece_number = StatisticsECE(Test_path)
    # print(ece_number, no_ece_number)

    # aver = ComputeV(Test_path)
    # print(aver)
    # Divide(cnn_folder, pre_train_folder)
    CopyData(pre_train_folder, cnn_folder)