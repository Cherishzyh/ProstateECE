import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from SSHProject.BasicTool.MeDIT.Normalize import NormalizeZ

from scipy.stats import mannwhitneyu


def WritePSACSV():
    case_list, age_list, psa_list = [], [], []
    for case_name in case_name_df.index:
        # case = case_name[:case_name.index('_slice')]
        case = case_name
        if case == 'DONG BAO SHENG':
            continue
        info = clinical_df.loc[case]
        if isinstance(info['PSA'], str):
            # if '＞' in info['PSA']:
            if '>' in info['PSA']:
                info['PSA'] = info['PSA'][1:]
        else:
            print('{} have no psa'.format(case))
            continue
        case_list.append(case_name)
        age_list.append(int(info['age']))
        psa_list.append(float(info['PSA']))

    age_norm = NormalizeZ(np.array(age_list))
    psa_norm = NormalizeZ(np.array(psa_list))

    age_psa_df = pd.DataFrame(list(map(list, zip(age_norm, psa_norm))), columns=['age', 'psa'], index=[case_list])
    age_psa_df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/Age&Psa_norm.csv')
    #


def WriteCSV(feature_name, clinical_df, case_list, save_path=''):
    feature_df = pd.DataFrame()

    feature_df['case'] = [case[: case.index('.npy')] for case in case_list]
    # feature_df['case'] = case_list
    if isinstance(feature_name, str):
        feature_name = [feature_name]

    for feature in feature_name:
        feature_list = []
        for case in case_list:
            case_name = case[:case.index('_slice')]
            # case_name = case
            if case_name in [case for case in clinical_df.index]:

                info = clinical_df.loc[case_name]

                try:
                    feature_list.append(float(info[feature]))

                except Exception:
                    print('the {} of {} cannot transform to float'.format(feature, case_name))
        if feature == 'pGs':
            feature_list = np.array(feature_list).clip(0, 4).tolist()
        feature_list = NormalizeZ(np.array(feature_list))

        feature_df[feature] = feature_list

    if save_path:
        feature_df.to_csv(save_path, index=False, encoding='gbk', mode='a+', header=False)

    return feature_df


def DrawDistributionHist(feature_list, feature_name, label_list):
    negative, positive = [], []
    for index in range(len(feature_list)):
        if label_list[index] == 1:
            positive.append(feature_list[index])
        else:
            negative.append(feature_list[index])
    print(mannwhitneyu(positive, negative))
    plt.title('{}'.format(feature_name))
    plt.hist(negative, color='b', alpha=0.5, bins=20, label='negative')
    plt.hist(positive, color='r', alpha=0.5, bins=20, label='positive')
    plt.legend()
    plt.show()


if __name__ == '__main__':

    JSPH_clinical_report_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/train_clinical.csv'
    case_csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ece.csv'

    case_list = os.listdir(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice/Test')
    if 'Test' in case_list:
        case_list.remove('Test')
        case_list.remove('DSR^dai shou rong_slice16.npy')

    clinical_df = pd.read_csv(JSPH_clinical_report_path, encoding='gbk',
                              usecols=['case', 'age', 'psa', 'bGs', 'core', 'b-NI'], index_col='case')

    # SUH_clinical_report_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/suh_clinical_supplement.csv'
    # case_csv_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/label.csv'
    #
    # case_list = os.listdir(r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice')
    #
    # clinical_df = pd.read_csv(SUH_clinical_report_path, encoding='gbk',
    #                           usecols=['case', 'age', 'PSA', '穿刺GS grade', 'core', 'b-NI'], index_col='case')
    #
    # case_name_df = pd.read_csv(case_csv_path, usecols=['case', 'ece'], index_col=['case'])
    WriteCSV(['age', 'psa', 'bGs', 'core', 'b-NI'], clinical_df, case_list,
             save_path=r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/FiveClinicalbGS.csv')



    #
    # label_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ece.csv'
    # csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/FiveClinical.csv'
    # label_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/label.csv'
    # csv_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/FiveClinical.csv'
    # feature_df = pd.read_csv(csv_path, index_col='case')
    # label_df = pd.read_csv(label_path, index_col='case')
    # age_list = feature_df['age']
    # psa_list = feature_df['psa']
    # pGs_list = feature_df['pGs']
    # core_list = feature_df['core']
    # b_NI_list = feature_df['b_NI']
    # label_list = label_df['ece']

    # age_list = []
    # psa_list = []
    # bGs_list = []
    # core_list = []
    # b_NI_list = []
    # label_list = []
    #
    # for case in os.listdir(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice/Test'):
    #     if case == 'Test' or case == 'DSR^dai shou rong_slice16.npy':
    #         continue
    #     else:
    #         # case = case[: case.index('_-_slice')]
    #         case = case[: case.index('.npy')]
    #
    #         age_list.append(feature_df.loc[case]['age'])
    #         psa_list.append(feature_df.loc[case]['psa'])
    #         bGs_list.append(feature_df.loc[case]['bGs'])
    #         core_list.append(feature_df.loc[case]['core'])
    #         b_NI_list.append(feature_df.loc[case]['b-NI'])
    #         label_list.append(label_df.loc[case]['ece'])
    #         # age_list.append(feature_df.loc[case]['age'])
    #         # psa_list.append(feature_df.loc[case]['PSA'])
    #         # pGs_list.append(feature_df.loc[case]['pGs'])
    #         # core_list.append(feature_df.loc[case]['core'])
    #         # b_NI_list.append(feature_df.loc[case]['b-NI'])
    #         # label_list.append(label_df.loc[case]['label'])
    #
    # DrawDistributionHist(age_list, 'Age in train', label_list)
    # DrawDistributionHist(psa_list, 'PSA in train', label_list)
    # DrawDistributionHist(bGs_list, 'bGs in internal test', label_list)
    # DrawDistributionHist(core_list, 'Core in train', label_list)
    # DrawDistributionHist(b_NI_list, 'b_NI in train', label_list)





