import os
import random
import shutil
import numpy as np
import pandas as pd

from BasicTool.MeDIT.SaveAndLoad import LoadH5
from random import shuffle

# from FilePath import *
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


def DivideDatabyCSV(csv_path):
    info = pd.read_csv(csv_path, index_col='CaseName')
    case_list = []
    label_list = []
    case_dict = {}
    for case in info.index:
        label_list.append(int(info.loc[case]['ECE']))
        case_list.append(case)
        case_dict[case] = int(info.loc[case]['ECE'])

    test_num = int(len(case_list) * 0.2)
    train_num = int((len(case_list) - test_num) * 0.8)
    val_num = len(case_list) - test_num - train_num

    print('total num: {}, train num: {}, val num: {}, test num: {}'.format(len(label_list), train_num, val_num, test_num))
    print('Positive Sample: {}'.format(sum(label_list)))
    print('Negative Sample: {}'.format(len(label_list)-sum(label_list)))

    shuffle(case_list)

    train_case, val_case, test_case = [], [], []
    train_label, val_label, test_label = [], [], []

    for case in case_list:
        if len(train_case) < train_num:
            train_case.append(str(case))
            train_label.append(case_dict[case])
        elif len(val_case) < val_num:
            val_case.append(str(case))
            val_label.append(case_dict[case])
        else:
            test_case.append(str(case))
            test_label.append(case_dict[case])

    print('Train: positive: {}, negative num: {}'.format(sum(train_label), len(train_label)-sum(train_label)))
    print(' Val : positive: {}, negative num: {}'.format(sum(val_label), len(val_label)-sum(val_label)))
    print('Test : positive: {}, negative num: {}'.format(sum(test_label), len(test_label)-sum(test_label)))

    train_df = pd.DataFrame(train_case).T
    val_df = pd.DataFrame(val_case).T
    test_df = pd.DataFrame(test_case).T
    train_df.to_csv(r'C:\Users\ZhangYihong\Desktop\train-name.csv')
    val_df.to_csv(r'C:\Users\ZhangYihong\Desktop\val-name.csv')
    test_df.to_csv(r'C:\Users\ZhangYihong\Desktop\test-name.csv')


def ChangeName():
    folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/kindey_slice'
    for case in os.listdir(folder):
        data = np.load(os.path.join(folder, case))
        if '000' in case:
            case_name = case[3:]
            np.save(os.path.join(folder, case_name), data)
            os.remove(os.path.join(folder, case))


def WriteCSV():
    # csv_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/RCC-ECE-New.CSV'
    name_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/test-name.csv'
    case_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/ct_slice'

    # csv_info = pd.read_csv(csv_path, index_col='CaseName')
    case_name_info = pd.read_csv(name_path).T
    case_name_info = case_name_info.drop(index='Unnamed: 0')

    case_name_list = []
    case_list = []

    for index in case_name_info.index:
        case_name_list.append(case_name_info.loc[index][0])

    for case in os.listdir(case_path):
        case_name = case[: case.index('_-_')]
        case_name_df = case[: case.index('.npy')]
        if case_name in str(case_name_list):
            case_list.append(case_name_df)

    df = pd.DataFrame(case_list).T
    df.to_csv('/home/zhangyihong/Documents/Kindey901/Kindey_npy/test-name-test.csv')


def WriteLabel():
    case_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/ct_slice'

    label_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/alltrain-name.csv'
    test_name_df = pd.read_csv(label_path, index_col=0).T
    test_case_list = np.squeeze(test_name_df.values).tolist()
    test_case_list = sorted(test_case_list)

    csv_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/RCC-ECE-New.CSV'
    csv_info = pd.read_csv(csv_path, dtype={0: str})
    csv_info.set_index(csv_info.columns.tolist()[0], inplace=True)

    case_list = []
    label_list = []

    for case in os.listdir(case_path):
        case_name = case[: case.index('_-_')]
        case_name_slice = case[: case.index('.npy')]
        # case_name in case_folder and case_name in label_
        if case_name_slice in test_case_list:
            pass
        else:
            if case_name in csv_info.index:
                case_list.append(case_name_slice)
                label_list.append(csv_info.loc[case_name]['ECE'])
            else:
                print(case_name)

    df = pd.DataFrame({'name': case_list, 'label': label_list})
    df.to_csv('/home/zhangyihong/Documents/Kindey901/Kindey_npy/test_label.csv', index=False)


if __name__ == '__main__':

    # b_value = ['0', '50', '700', '750', '1400', '1500']
    # case_list = os.listdir(process_folder)
    # for case in case_list:
    #     b = GetBValue(case, b_value)
    #     if isinstance(b, str):
    #         if float(b) < 1200:
    #             print(case)

    # csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/csv/ece.csv'
    # cnn_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/RoiSlice/Test'
    # pre_train_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/RoiSlice/PreValid'
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
    # CopyData(pre_train_folder, cnn_folder)
    # test_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\Test\AdcSlice'
    # source_folder = r'Y:\ZYH\ResampleData'
    # des_folder = r'Y:\ZYH\Test'
    # for case in os.listdir(test_folder):
    #     case_name = case[:case.index('_slice')]
    #     shutil.move(os.path.join(source_folder, case_name), os.path.join(des_folder, case_name))


    # csv_path = r'C:\Users\ZhangYihong\Desktop\RCC-ECE-New.CSV'
    # DivideDatabyCSV(csv_path)
    # ChangeName()
    # WriteCSV()
    WriteLabel()