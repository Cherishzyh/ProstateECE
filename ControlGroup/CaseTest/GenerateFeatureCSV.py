import pandas as pd
import numpy as np

from BasicTool.MeDIT.Normalize import Normalize01


def GetCSVListByCase(pd, feature, case):
    feature_list = []
    case_list = []
    for index in pd.index:
        if index in case:
            feature_list.append(pd.loc[index][feature])
            case_list.append(index)
    return case_list, feature_list


def GetCSVList(pd, feature):
    feature_list = []
    case_list = []
    for index in pd.index:
        feature_list.append(pd.loc[index][feature])
        case_list.append(pd.loc[index]['case'])
    return case_list, feature_list


# pred, MRECE
def GenerateCSV(pred_csv_path, clinical_csv_path, feature_list, coding='one_hot'):
    pred_df = pd.read_csv(pred_csv_path)
    clinical_df = pd.read_csv(clinical_csv_path, index_col='case')


    case, label = GetCSVList(pred_df, 'Label', )
    _, pred = GetCSVList(pred_df, 'PAGNet')
    # pred = np.clip(np.array(pred), 0.01, 0.99)
    # pred_log = Normalize01(np.log(pred) - np.log(1-pred)).tolist()

    feature_dict = {'case': case, 'label': label, 'PAGNet': pred}

    for feature in feature_list:
        _, feature_index = GetCSVListByCase(clinical_df, feature, case)

        if feature == 'b-NI' or feature == 'MR_ECE':
            feature_dict[feature] = feature_index

        elif feature == 'bGs':
            for index in range(len(feature_index)):
                if feature_index[index] == 1:
                    feature_index[index] = [1, 0, 0, 0]
                elif feature_index[index] == 2:
                    feature_index[index] = [0, 1, 0, 0]
                elif feature_index[index] == 3:
                    feature_index[index] = [0, 0, 1, 0]
                else:
                    feature_index[index] = [0, 0, 0, 1]
            feature_dict[feature+'_4'] = np.array(feature_index)[:, 0].tolist()
            feature_dict[feature + '_3'] = np.array(feature_index)[:, 1].tolist()
            feature_dict[feature + '_2'] = np.array(feature_index)[:, 2].tolist()

        elif feature == 'ECE stage bj' or feature == 'ECE stage hy':
            if coding == 'one_hot':
                for index in range(len(feature_index)):
                    if feature_index[index] == 0:
                        feature_index[index] = [1, 0, 0, 0]
                    elif feature_index[index] == 1:
                        feature_index[index] = [0, 1, 0, 0]
                    elif feature_index[index] == 2:
                        feature_index[index] = [0, 0, 1, 0]
                    else:
                        feature_index[index] = [0, 0, 0, 1]
                feature_dict[feature + '_3'] = np.array(feature_index)[:, 3].tolist()
                feature_dict[feature + '_2'] = np.array(feature_index)[:, 2].tolist()
                feature_dict[feature + '_1'] = np.array(feature_index)[:, 1].tolist()
                feature_dict[feature + '_0'] = np.array(feature_index)[:, 0].tolist()
            elif coding == 'ordinal':
                for index in range(len(feature_index)):
                    if feature_index[index] == 0:
                        feature_index[index] = [0, 0, 0]
                    elif feature_index[index] == 1:
                        feature_index[index] = [0, 0, 1]
                    elif feature_index[index] == 2:
                        feature_index[index] = [0, 1, 1]
                    else:
                        feature_index[index] = [1, 1, 1]
                feature_dict[feature + '_3'] = np.array(feature_index)[:, 0].tolist()
                feature_dict[feature + '_2'] = np.array(feature_index)[:, 1].tolist()
                feature_dict[feature + '_1'] = np.array(feature_index)[:, 2].tolist()

        else:
            feature_index = Normalize01(np.array(feature_index)).tolist()
            feature_dict[feature] = feature_index

    return feature_dict
    # feature_matrix = pd.DataFrame()
    # feature_matrix.to_csv(save_path)

if __name__ == '__main__':
    test_path = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\test.csv'
    # train_path = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\train.csv'
    # external_path = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh.csv'

    # test_clinical_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\test_clinical.csv'
    # train_clinical_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\train_clinical.csv'

    clinical_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-JSPH-clinical_report.csv'
    # clinical_path = r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\suh_clinical_supplement.csv'
    # train_clinical_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\train_clinical.csv'

    feature_list = ['ECE stage bj', 'ECE stage hy']

    # feature_dict = GenerateCSV(test_path, test_clinical_path, feature_list)
    # feature_matrix = pd.DataFrame(feature_dict)
    # feature_matrix.to_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\CaseBased_NoSigmoid\test.csv')
    # print()

    feature_dict = GenerateCSV(test_path, clinical_path, feature_list, coding='ordinal')
    feature_matrix = pd.DataFrame(feature_dict)
    feature_matrix.to_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\Grade\PAGNet+grade_test_ordinal.csv')
    print()
