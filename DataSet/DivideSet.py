import os
import random
import shutil
import numpy as np

from MeDIT.SaveAndLoad import LoadH5

from NPYFilePath import *
# 正负样本比例1:3
# b值800的在test

def DivideData(folder, des_folder):
    info = pd.read_csv(r'/home/zhangyihong/Documents/ProstateECE/NPY/ECE-ROI.csv'
                       , usecols=['case', 'pECE'], index_col=['case'])
    case_list = os.listdir(folder)
    random.shuffle(case_list)
    ece_number = 0
    no_ece_number = 0
    for case in case_list:
        case_path = os.path.join(folder, case)
        des_path = os.path.join(des_folder, case)
        if not os.path.isdir(case_path):
            case_name = case[:case.index('_slice')]
            ece = info.loc[case_name, 'pECE']

            if all(ece == np.array([1])):
                if no_ece_number == 15:
                    continue
                else:
                    no_ece_number += 1
                    shutil.move(case_path, des_path)
            elif all(ece == np.array([0])):
                if ece_number == 45:
                    continue
                else:
                    ece_number += 1
                    shutil.move(case_path, des_path)
            else:
                print(case)


def DivideCopy(des_folder, folder):
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
    case_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/AdcSlice/PreTrain'
    case_list = os.listdir(case_folder)
    for case in case_list:
        case_path = os.path.join(folder, case)
        des_case_path = os.path.join(des_folder, case)
        shutil.move(case_path, des_case_path)


if __name__ == '__main__':
    import pandas as pd
    train_path = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/DwiSlice/Train'
    pre_train_path = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain/DwiSlice/PreTrain'

    # DivideData(train_path, pre_train_path)
    # DivideCopy(pre_train_path, train_path)
    CopyData(pre_train_path, train_path)