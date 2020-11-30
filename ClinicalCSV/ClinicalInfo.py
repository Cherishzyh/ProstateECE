import os
import pandas as pd

def CaseClinicalCSV(train_list, test_list):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for index in clinical_info.index:
        case_name = clinical_info.loc[index]['case']
        if case_name == 'DSR^dai shou rong':
            continue
        if case_name in train_list:
            train_df = train_df.append(clinical_info.loc[index], ignore_index=True)
            # pass
        elif case_name in test_list:
            test_df = test_df.append(clinical_info.loc[index], ignore_index=True)
            # pass
    train_df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\train_clinical.csv', encoding='gbk')
    test_df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\test_clinical.csv', encoding='gbk')
    print()


if __name__ == '__main__':
    train_list = os.listdir(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\AdcSlice')
    train_list = [case[: case.index('_slice')] for case in train_list]
    #
    test_list = os.listdir(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\Test\AdcSlice')
    test_list = [case[: case.index('_slice')] for case in test_list]
    #
    clinical_info = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-ROI.csv', encoding='gbk')
    clinical_info = clinical_info.drop(['name'], axis=1)

    # suh_list = os.listdir(r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\ProstateSlice')
    # suh_list = [case[: case.index('_-_slice')] for case in suh_list]
    # suh_info = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\SUH_ECE_clinical-report.csv', encoding='gbk')
    # suh_info = suh_info.drop(['姓名'], axis=1)

    CaseClinicalCSV(train_list, test_list)