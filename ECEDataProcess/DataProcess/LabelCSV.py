import pandas as pd
import numpy as np
import os


def test():
    ece_path = r'X:\StoreFormatData\ProstateCancerECE\ECE-ROI.csv'
    data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    store_csv_path = r''

    info = pd.read_csv(ece_path, usecols=['case', 'pECE'], index_col=['case'])

    case_list = os.listdir(data_folder)
    for case in case_list:
        ece = info.loc[case, 'pECE']

        if ece == 0:
            ece = np.array([0, 1], dtype=np.uint8)
            label_df = pd.DataFrame({'3': [0], '2': [0], '1': [0]}, index=[case])
            # label_df = pd.DataFrame([[0], [0], [0], [1]], index=[case_name])
            label_df.to_csv(store_csv_path, header=False, mode='a')
        elif ece == 1:
            ece = np.array([1, 0], dtype=np.uint8)


def DataDivide():
    case_list = os.listdir(r'X:\CNNFormatData\ProstateCancerECE\NPYMaxPred\Test\AdcSlice')
    case_list = [case[:case.index('_-_')] for case in case_list]
    new_feature = pd.DataFrame({'CaseName': case_list})
    new_feature.to_csv(r'C:\Users\ZhangYihong\Desktop\test\MultiFeature\test_ref.csv', index=False)
DataDivide()