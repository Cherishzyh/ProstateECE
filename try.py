import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from BasicTool.MeDIT.Statistics import BinaryClassification
from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.Visualization import Imshow3DArray

from copy import deepcopy

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn import metrics

import matplotlib.pylab as mpl

from pylab import *


def ComputeAUC():
    csv_path = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh_right.csv'
    df = pd.read_csv(csv_path, index_col='case')
    pagnet_list = []
    resnext_list = []
    label = []
    pagnet_c_list = []
    for index in df.index:
        label.append(int(df.loc[index]['label']))
        pagnet_list.append(df.loc[index]['PAGNet'])
        resnext_list.append(df.loc[index]['ResNeXt'])
        pagnet_c_list.append(df.loc[index]['PAGNet + C'])
    from BasicTool.MeDIT.Statistics import BinaryClassification
    bc = BinaryClassification()
    bc.Run(pagnet_list, label)
    bc.Run(resnext_list, label)
    bc.Run(pagnet_c_list, label)


def WriteAllResultCSV():
    csv_path_hy = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\hy_suh.csv'
    csv_pagnet = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh.csv'
    csv_resnext = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\ResNeXt\suh.csv'
    csv_pagnet_c = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet+c\suh.csv'

    # case, label, pagnet, resnext, pagnet+c,
    # model+ECE grade hy, model+ECE grade bj, hy pred, bj pred, RISK, PI-RADS, Dmax,

    df_hy = pd.read_csv(csv_path_hy, index_col='case')

    df_pagent = pd.read_csv(csv_pagnet, index_col='case')
    df_resnext = pd.read_csv(csv_resnext, index_col='case')
    df_pagent_c = pd.read_csv(csv_pagnet_c, index_col='case')

    case_list, label_list, pagnet_list, resnext_list, pagnet_c_list = [], [], [], [], []
    model_ECE_hy, model_ECE_bj, hy_pred, bj_pred, RISK, PI_RADS, Dmax = [], [], [], [], [], [], []

    for case in df_hy.index:
        case_list.append(case)
        label_list.append(df_pagent.loc[case]['Label'])
        pagnet_list.append(df_pagent.loc[case]['PAGNet'])
        resnext_list.append(df_resnext.loc[case]['Pred'])
        pagnet_c_list.append(df_pagent_c.loc[case]['Pred'])
        model_ECE_hy.append(df_hy.loc[case]['model+ECE grade hy'])
        model_ECE_bj.append(df_hy.loc[case]['model+ECE grade bj'])
        hy_pred.append(df_hy.loc[case]['hy pred'])
        bj_pred.append(df_hy.loc[case]['bj pred'])
        RISK.append(df_hy.loc[case]['RISK'])
        PI_RADS.append(df_hy.loc[case]['PIRADS'])
        Dmax.append(df_hy.loc[case]['D-max'])

    dict = {'case': case_list, 'label': label_list, 'PAGNet': pagnet_list, 'ResNeXt': resnext_list, 'PAGNet + C': pagnet_c_list,
            'model+ECE grade hy': model_ECE_hy, 'model+ECE grade bj':model_ECE_bj,
            'hy pred': hy_pred, 'bj pred': bj_pred,
            'RISK': RISK, 'PI-RADS':PI_RADS, 'Dmax':Dmax}
    df = pd.DataFrame(dict)
    df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh_right.csv')

    right_csv = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh_right.csv'
    right = pd.read_csv(right_csv, index_col='case')
    for index in right.index:
        if abs(right.loc[index]['PAGNet'] - df_pagent.loc[index]['PAGNet']) > 1e-8:
            print(index, 'PAGNet')
        if abs(right.loc[index]['ResNeXt'] - df_resnext.loc[index]['Pred']) > 1e-8:
            print(index, 'ResNext')
        if abs(right.loc[index]['PAGNet + C'] - df_pagent_c.loc[index]['Pred']) > 1e-8:
            print(index, 'PAGNet+C')


def AllResultCSVSupplement():
    csv_path_3D = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh_right.csv'

    csv_pagnet_2d = r'X:\CNNFormatData\ProstateCancerECE\Result\PAGNet_suh.csv'
    csv_resnext_2d = r'X:\CNNFormatData\ProstateCancerECE\Result\ResNeXt_suh.csv'

    model_grade_hy_2d = r'C:\Users\ZhangYihong\Desktop\JMRI\feature\Grade2d\HY\suh\None\PCC\Relief_5\LR\test_prediction.csv'
    model_grade_bj_2d = r'C:\Users\ZhangYihong\Desktop\JMRI\feature\Grade2d\BJ\suh\None\PCC\Relief_5\LR\test_prediction.csv'

    # npy
    csv_pagnet_c_2d = r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeaturebGS\suh.csv'

    df_all = pd.read_csv(csv_path_3D, index_col='case')

    df_pagent = pd.read_csv(csv_pagnet_2d, index_col='case')
    df_resnext = pd.read_csv(csv_resnext_2d, index_col='case')

    df_model_grade_hy = pd.read_csv(model_grade_hy_2d, index_col='case')
    df_model_grade_bj = pd.read_csv(model_grade_bj_2d, index_col='case')

    df_pagent_c = pd.read_csv(csv_pagnet_c_2d, index_col='case')
    #
    case_list, label_list, pagnet_list, resnext_list, model_grade_hy, model_grade_bj, pagnet_c = [], [], [], [], [], [], []

    for case in df_all.index:
        case_list.append(case)
        label_list.append(df_all.loc[case]['label'])

        pagnet_list.append(df_pagent.loc[case]['Pred'])
        resnext_list.append(df_resnext.loc[case]['Pred'])

        model_grade_hy.append(df_model_grade_hy.loc[case]['Pred'])
        model_grade_bj.append(df_model_grade_bj.loc[case]['Pred'])

        pagnet_c.append(df_pagent_c.loc[case]['Pred'])

    df_all['PAGNet 2D'] = pagnet_list
    df_all['ResNeXt 2D'] = resnext_list
    df_all['PAGNet + C 2D'] = pagnet_c
    df_all['model+ECE grade hy 2D'] = model_grade_hy
    df_all['model+ECE grade bj 2D'] = model_grade_bj

    del df_all['Unnamed: 0']
    df_all.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\suh.csv')


def Model2Dvs3D():
    from BasicTool.MeDIT.Statistics import MyWilcoxon, BinaryClassification
    from BasicTool.MeDIT.Base.DeLongTest import DeLongTest
    pagnet_3D_csv = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\test.csv'
    pagnet_2D_csv = r'X:\CNNFormatData\ProstateCancerECE\Result\PAGNet_test.csv'
    df_3d = pd.read_csv(pagnet_3D_csv, index_col='case')
    df_2d = pd.read_csv(pagnet_2D_csv, index_col='case')
    case_list = []
    label = []
    pagnet_2D = []
    pagnet_3D = []
    for case in df_2d.index:
        if df_3d.loc[case]['Label'] == df_2d.loc[case]['Label']:
            case_list.append(case)
            label.append(int(df_2d.loc[case]['Label']))
            pagnet_2D.append(df_2d.loc[case]['Pred'])
            pagnet_3D.append(df_3d.loc[case]['Pred'])
        else:
            print(case)

    dl = DeLongTest()
    # bc = BinaryClassification()

    # print(MyWilcoxon(pagnet_2D, pagnet_3D))
    # print(dl.DelongRocTest(np.array(label).astype(int), np.array(pagnet_3D), np.array(pagnet_2D)))
    aver = ((np.array(pagnet_3D) + np.array(pagnet_2D))/2).tolist()

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, _ = roc_curve(label, pagnet_2D)
    auc = roc_auc_score(label, pagnet_2D)
    plt.plot(fpn, sen, label='PAGNet 2D: {:.3f}'.format(auc))

    fpn, sen, _ = roc_curve(label, pagnet_3D)
    auc = roc_auc_score(label, pagnet_3D)
    plt.plot(fpn, sen, label='PAGNet 3D: {:.3f}'.format(auc))

    fpn, sen, _ = roc_curve(label, aver)
    auc = roc_auc_score(label, aver)
    plt.plot(fpn, sen, label='Average: {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')

    plt.show()

    # bc.Run(((np.array(pagnet_3D) + np.array(pagnet_2D))/2).tolist(), label)
    # info = pd.DataFrame({'case': case_list, 'label': label, 'max roi': pagnet_2D, 'max pred': pagnet_3D})
    # info.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\ModelCompare.csv')


def ComputeMetric():
    store_folder = r'C:\Users\ZhangYihong\Desktop\JMRI\feature\Grade2d\PAGNet+ECEGrade\bj_result'
    train_csv = os.path.join(store_folder, 'train_prediction.csv')
    test_csv = os.path.join(store_folder, 'test_prediction.csv')
    suh_csv = os.path.join(store_folder, 'suh_prediction.csv')

    df_train = pd.read_csv(train_csv, index_col='case')
    df_test = pd.read_csv(test_csv, index_col='case')
    df_suh = pd.read_csv(suh_csv, index_col='case')

    # label_list = []
    for df in [df_train, df_test, df_suh]:
        label_list = []
        pred_list = []
        for case in df.index:
            label_list.append(int(df.loc[case]['Label']))
            pred_list.append(df.loc[case]['Pred'])
        bc = BinaryClassification()
        bc.Run(pred_list, label_list)


def SelectCase():
    csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ece.csv'
    label_df = pd.read_csv(csv_path, index_col='case')

    t2_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/T2Slice/Test'
    n_list, p_list = [], []
    for index, case in enumerate(os.listdir(t2_folder)):
        label = label_df.loc[case[: case.index('.npy')]]['ece']
        if len(p_list) > 10 and len(n_list) > 10:
            break
        if label == 1.:
            if len(p_list) >= 10:
                continue
            else:
                p_list.append(case)
                print(label, case)
        elif label == 0.:
            if len(n_list) >= 10:
                continue
            else:
                n_list.append(case)
                print(label, case)

    p_list.extend(n_list)

    return p_list


def SelectCaseEx():
    csv_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/label.csv'
    label_df = pd.read_csv(csv_path, index_col='case')

    t2_folder = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice'
    n_list, p_list = [], []
    for index, case in enumerate(os.listdir(t2_folder)):
        label = label_df.loc[case[: case.index('_-_')]]['label']
        if len(p_list) > 10 and len(n_list) > 10:
            break
        if label == 1.:
            if len(p_list) >= 10:
                continue
            else:
                p_list.append(case)
                print(label, case)
        elif label == 0.:
            if len(n_list) >= 10:
                continue
            else:
                n_list.append(case)
                print(label, case)

    p_list.extend(n_list)

    return p_list


def CutOff(list):
    length = len(list)
    cut_num = int(length * 0.05)
    new_list = list[cut_num: length-cut_num]
    print('orignal len{}, cut len{}'.format(length, len(new_list)))
    return new_list


def HistDistribution():
    case_in = SelectCase()
    case_ex = SelectCaseEx()

    df = pd.DataFrame({'case_in': case_in, 'case_ex': case_ex})
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/select.csv', index=False)

    t2_folder_in = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/T2Slice/Test'
    pro_folder_in = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ProstateSlice/Test'
    pca_folder_in = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/RoiSlice/Test'

    t2_folder_ex = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/T2Slice'
    pro_folder_ex = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/ProstateSlice'
    pca_folder_ex = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/PCaSlice'

    t2_list_in, t2_list_ex = [], []
    pro_list_in, pro_list_ex = [], []
    pca_list_in, pca_list_ex = [], []

    for case in case_in:
        t2_path = os.path.join(t2_folder_in, case)
        pro_path = os.path.join(pro_folder_in, case)
        pca_path = os.path.join(pca_folder_in, case)

        t2 = np.squeeze(np.load(t2_path))
        pro = np.squeeze(np.load(pro_path))
        pca = np.squeeze(np.load(pca_path))

        t2_list_in.extend(t2.flatten().tolist())

        pro_list_in.extend(t2[pro == 1].tolist())

        pca_list_in.extend(t2[pca == 1].tolist())

    for case in case_ex:
        t2_path = os.path.join(t2_folder_ex, case)
        pro_path = os.path.join(pro_folder_ex, case)
        pca_path = os.path.join(pca_folder_ex, case)

        t2 = np.squeeze(np.load(t2_path))
        pro = np.squeeze(np.load(pro_path))
        pca = np.squeeze(np.load(pca_path))

        t2_list_ex.extend(t2.flatten().tolist())
        pro_list_ex.extend(t2[pro == 1].tolist())
        pca_list_ex.extend(t2[pca == 1].tolist())

    t2_list_in = CutOff(sorted(t2_list_in))
    pro_list_in = CutOff(sorted(pro_list_in))
    pca_list_in = CutOff(sorted(pca_list_in))
    t2_list_ex = CutOff(sorted(t2_list_ex))
    pro_list_ex = CutOff(sorted(pro_list_ex))
    pca_list_ex = CutOff(sorted(pca_list_ex))

    plt.title('Distribution of T2')
    plt.hist(t2_list_in, alpha=0.5, bins=50, density=True, label='internal')
    plt.hist(t2_list_ex, alpha=0.5, bins=50, density=True, label='external')
    plt.legend()
    plt.show()

    plt.title('Distribution of prostate in T2')
    plt.hist(pro_list_in, alpha=0.5, bins=50, density=True, label='internal')
    plt.hist(pro_list_ex, alpha=0.5, bins=50, density=True, label='external')
    plt.legend()
    plt.show()

    plt.title('Distribution of PCa in T2')
    plt.hist(pca_list_in, alpha=0.5, bins=50, density=True, label='internal')
    plt.hist(pca_list_ex, alpha=0.5, bins=50, density=True, label='external')
    plt.legend()
    plt.show()


def test():
    csv_path = r'C:\Users\ZhangYihong\Desktop\test.csv'
    csv_info = pd.read_csv(csv_path, dtype={0: str})
    csv_info.set_index(csv_info.columns.tolist()[0], inplace=True)

    print(csv_info.index)


if __name__ == '__main__':

    # data = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\AdcSlice\BI JUN_slice11.npy')
    # print(data.shape)
    # train_label = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeature\SUH_dis&five_label.npy')
    # train_bj = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\NPY\SUH_bj_result.npy')
    # train_hy = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\NPY\SUH_hy_result.npy')
    # train_dis_ff = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeature\SUH_dis&five_pred.npy')
    # train_nodis_ff = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeature\SUH_nodis&five_pred.npy')
    # train_dis_lf = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeatureLF\SUH_FiveFeature_pred.npy')
    # train_nodis_lf = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeatureLF\SUH_nodis&fivefeaturepred.npy')
    #
    # train_df = pd.DataFrame({'label': train_label, 'attention map, age, psa, pGS, core, b-NI in first fc': train_dis_ff,
    #                          'age, psa, pGS, core, b-NI in first fc': train_nodis_ff,
    #                          'attention map, age, psa, pGS, core, b-NI in last fc': train_dis_lf,
    #                          'age, psa, pGS, core, b-NI in last fc': train_nodis_lf,
    #                          'hy pred': train_hy,
    #                          'bj pred': train_bj})
    # train_df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\SUH_ModelVSClinical_AddFiveFeature.csv')
    #
    # # train_label = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\NPY\SUH_label.npy')
    # # clinical_df = pd.DataFrame({'label': train_label, 'hy pred': train_hy, 'bj pred': train_bj})
    # # clinical_df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\SUH_ModelVSClinical_AddAge&PSA_train_clinical.csv')
    # print()


    # train_label = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeaturebGS\testJSPH_dis&five_label.npy')
    # train_dis_ff = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeaturebGS\testJSPH_dis&five_pred.npy')
    # train_dis_lf = np.load(r'X:\CNNFormatData\ProstateCancerECE\Result\AddClinicalNPY\FiveFeatureLFbGS\testJSPH_FiveFeature_pred.npy')
    #
    # train_df = pd.DataFrame({'label': train_label,
    #                          'attention map, age, psa, pGS, core, b-NI in first fc': train_dis_ff,
    #                          'attention map, age, psa, pGS, core, b-NI in last fc': train_dis_lf})
    # train_df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\JSPH_AddFiveFeature_test.csv')


    # test_clinical = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\train_clinical.csv'
    # model_pred = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\train.csv'
    #
    # test_clinical_df = pd.read_csv(test_clinical)
    # model_pred_df = pd.read_csv(model_pred)
    # for index in test_clinical_df.index:
    #     if test_clinical_df.loc[index]['case'] != model_pred_df.loc[index]['case']:
    #         print(index)
    # train_1 = r'C:\Users\ZhangYihong\Desktop\PAGNet+ECE_Grade\GradeBybj\suh\train_prediction.csv'
    # train_2 = r'C:\Users\ZhangYihong\Desktop\PAGNet+ECE_Grade\GradeBybj\test\train_prediction.csv'
    # train_1_df = pd.read_csv(train_1)
    # train_2_df = pd.read_csv(train_2)
    # for index in train_1_df.index:
    #     train_1_pred = train_1_df.loc[index]['Pred']
    #     train_2_pred = train_2_df.loc[index]['Pred']
    #     print(train_1_pred == train_2_pred)
    # attentiom_map_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/DistanceMap'
    # save_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/BinaryAttentionMap'
    # # attentiom_map_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/DistanceMap'
    # # save_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/BinaryAttentionMap'
    # for case in os.listdir(attentiom_map_path):
    #     if case == 'Test':
    #         pass
    #     else:
    #         atten_map = np.load(os.path.join(attentiom_map_path, case))
    #         binary_atten = np.where(atten_map >= 0.1, 1, 0)
    #         np.save(os.path.join(save_path, case), binary_atten)
        # atten_map_binary = np.load(os.path.join(save_path, case))
        # atten_map = np.load(os.path.join(attentiom_map_path, case))

    # ComputeMetric()
    # Model2Dvs3D()
    # train_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\train.csv')
    # label, pred = [], []
    # for index in train_df.index:
    #     label.append(train_df.loc[index]['Label'])
    #     pred.append(train_df.loc[index]['Pred'])
    # bc = BinaryClassification()
    # bc.Run(pred, label)

    #
    # DrawScatter()


    # csv_path = r'C:\Users\ZhangYihong\Desktop\SUH-JSPHTest.csv'
    # df = pd.read_csv(csv_path, index_col='case')
    # label_list, pagnet_list, hy_list, bj_list, pagnet_hy_list, pagnet_bj_list = [], [], [], [], [], []
    # for case in df.index:
    #     if float(df.loc[case]['RISK']) == 3.:
    #         label_list.append(int(df.loc[case]['label']))
    #         pagnet_list.append(float(df.loc[case]['PAGNet']))
    #         hy_list.append(int(df.loc[case]['hy pred']))
    #         bj_list.append(int(df.loc[case]['bj pred']))
    #         pagnet_hy_list.append(int(df.loc[case]['hy PAGNet']))
    #         pagnet_bj_list.append(int(df.loc[case]['bj PAGNet']))
    #
    # bc.Run(pagnet_list, label_list)
    # bc.Run(hy_list, label_list)
    # bc.Run(bj_list, label_list)
    # bc.Run(pagnet_hy_list, label_list)
    # bc.Run(pagnet_bj_list, label_list)

    # PAGNet
    # pred = np.zeros_like(label_list)
    # pred[np.array(pagnet_list) >= 2] = 1
    # C = metrics.confusion_matrix(label_list, pagnet_list)
    # TP, FP, FN, TN = C[1, 1], C[0, 1], C[1, 0], C[0, 0]
    # print(TP, FP, FN, TN)
    #
    # # HY
    # pred = np.zeros_like(label_list)
    # pred[np.array(hy_list) >= 2] = 1
    # C = metrics.confusion_matrix(label_list, pred)
    # TP, FP, FN, TN = C[1, 1], C[0, 1], C[1, 0], C[0, 0]
    # print(TP, FP, FN, TN)
    #
    # # BJ
    # pred = np.zeros_like(label_list)
    # pred[np.array(bj_list) >= 2] = 1
    # C = metrics.confusion_matrix(label_list, pred)
    # TP, FP, FN, TN = C[1, 1], C[0, 1], C[1, 0], C[0, 0]
    # print(TP, FP, FN, TN)
    #
    # # HY PAGNet
    # pred = np.zeros_like(label_list)
    # pred[np.array(pagnet_hy_list) >= 2] = 1
    # C = metrics.confusion_matrix(label_list, pred)
    # TP, FP, FN, TN = C[1, 1], C[0, 1], C[1, 0], C[0, 0]
    # print(TP, FP, FN, TN)
    #
    # # BJ PAGNet
    # pred = np.zeros_like(label_list)
    # pred[np.array(pagnet_bj_list) >= 1] = 1
    # C = metrics.confusion_matrix(label_list, pred)
    # TP, FP, FN, TN = C[1, 1], C[0, 1], C[1, 0], C[0, 0]
    # print(TP, FP, FN, TN)


    ######################################Show the Roi#################################################################
    # data_before = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide'
    # data_after = r'X:\CNNFormatData\ProstateCancerECE\NPYMaxPred\Train'
    # list_before = sorted(os.listdir(os.path.join(data_before, 'PCaSlice')))
    # list_before.remove('DSR^dai shou rong_slice16.npy')
    # list_after = sorted(os.listdir(os.path.join(data_after, 'PCaSlice')))
    #
    # pca_diff_list = []
    # for index in range(len(os.listdir(os.path.join(data_after, 'T2Slice')))):
    #     case_before = list_before[index]
    #     case_after = list_after[index]
    #     t2_before = np.squeeze(np.load(os.path.join(data_before, 'T2Slice/'+case_before)))
    #     t2_after = np.squeeze(np.load(os.path.join(data_after, 'T2Slice/'+case_after)))
    #     pca_before = np.squeeze(np.load(os.path.join(data_before, 'PCaSlice/'+case_before)))
    #     pca_after = np.squeeze(np.load(os.path.join(data_after, 'PCaSlice/'+case_after)))
    #     pro_before = np.squeeze(np.load(os.path.join(data_before, 'ProstateSlice/'+case_before)))
    #     pro_after = np.squeeze(np.load(os.path.join(data_after, 'ProstateSlice/'+case_after)))
    #     pca_diff_list.append(np.sum(pca_after) - np.sum(pca_before))
    #     plt.subplot(121)
    #     plt.imshow(t2_before, cmap='gray')
    #     plt.contour(pca_before, colors='r')
    #     plt.contour(pro_before, colors='y')
    #     plt.subplot(122)
    #     plt.imshow(t2_after, cmap='gray')
    #     plt.contour(pca_after, colors='r')
    #     plt.contour(pro_after, colors='y')
    #     plt.show()

    # plt.hist(pca_diff_list, bins=20)
    # plt.show()

    # ece_csv = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\label.csv'
    # info = pd.read_csv(ece_csv)
    # case_list = []
    # neg_list = []
    # pos_list = []
    #
    # for index in info.index:
    #     case_name = info.loc[index]['case']
    #     case, slice = case_name.split('_')
    #     case_list.append(case)
    #     neg_list.append(int(info.loc[index]['Negative']))
    #     pos_list.append(int(info.loc[index]['Positive']))
    # new_info = pd.DataFrame({'case': case_list, 'Negative': neg_list, 'Positive': pos_list})
    # new_info.to_csv(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\label_new.csv', index=False)

    # from BasicTool.MeDIT.SaveAndLoad import LoadH5AllTag
    # data = LoadH5AllTag(r'X:\CNNFormatData\PzTzSegment_FormatH5\input0_output0\2019-CA-formal-BAO TONG-slicer_index_0.h5')
    # print(data)
    #
    # test_df = pd.DataFrame(test_name)
    # train_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/train_name.csv', index=False)

    # data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Train/T2Slice'
    # data_list = os.listdir(data_folder)
    # data_list = [case[: case.index('_-_slice')] for case in data_list]
    # shuffle(data_list)
    # train_list = []
    # val_list = []
    # for case in data_list:
    #     if len(train_list) < 477:
    #         train_list.append(case)
    #     else:
    #         val_list.append(case)
    #
    # print(len(train_list), len(val_list))
    #
    # train_df = pd.DataFrame(train_list)
    # train_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/train_name_basemodel.csv', index=False)
    #
    # val_df = pd.DataFrame(val_list)
    # val_df.T.to_csv(r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/val_name_basemodel.csv', index=False)



    # bc = BinaryClassification()
    # label = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_label.npy').tolist()
    # pred = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_dis_result.npy').tolist()
    # bc.Run(pred, label)

    # HistDistribution()

    ####################################################################################################################
    # t2_nii_path = r'X:\PrcoessedData\ProstateCancerECE_SUH\SHEN ZHI XIN\t2_5x5.nii'
    # prostate_path = r'X:\PrcoessedData\ProstateCancerECE_SUH\SHEN ZHI XIN\prostate_roi_5x5.nii.gz'
    # pca_path = r'X:\PrcoessedData\ProstateCancerECE_SUH\SHEN ZHI XIN\pca_roi_5x5.nii.gz'
    # _, t2, _ = LoadImage(t2_nii_path)
    # _, roi, _ = LoadImage(prostate_path)
    # _, pca, _ = LoadImage(pca_path)
    #
    # Imshow3DArray(Normalize01(t2), roi=[Normalize01(roi), Normalize01(pca)])
    ####################################################################################################################

    # AddLabel()
    # ConcatCSV()
    # Show()


    ##########################################show image data###########################################################
    # _, ct, _ = LoadImage(r'X:\RawData\Kindey901_new\601548\data.nii.gz')
    # _, gland, _ = LoadImage(r'X:\RawData\Kindey901_new\601548\only_kidney_roi_lq.nii.gz')
    # _, roi, _ = LoadImage(r'X:\RawData\Kindey901_new\601548\roi.nii.gz')
    # Imshow3DArray(Normalize01(ct), roi=[Normalize01(gland), Normalize01(roi)])

    ##########################################write label###############################################################
    # csv_path = r'C:\Users\ZhangYihong\Desktop\RCC-ECE-New.CSV'
    # name_path = r'C:\Users\ZhangYihong\Desktop\alltrain-name.csv'
    # case_path = r'X:\CNNFormatData\Kindey_npy\ct_slice'
    #
    # csv_info = pd.read_csv(csv_path, index_col='CaseName')
    # case_info = pd.read_csv(name_path).T
    # case_info = case_info.drop(index='Unnamed: 0')
    #
    # case_list = []
    # label_list = []
    #
    # case_list_df = []
    #
    # for index in case_info.index:
    #     case_list.append(case_info.loc[index][0])
    #
    # for case in os.listdir(case_path):
    #     case_name = case[: case.index('_-_')]
    #     case_name_df = case[: case.index('.npy')]
    #     if case_name in str(case_list):
    #         case_list_df.append(case_name_df)
    #         label_list.append(csv_info.loc[int(case_name)]['ECE'])
    #
    # df = pd.DataFrame({'name': case_list_df, 'label': label_list})
    # df.to_csv('C:/Users/ZhangYihong/Desktop/alltrain_label.csv')


    ##############################################check data############################################################

    # standard_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/atten_image'
    # target_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/atten_slice'
    #
    # standard_list = os.listdir(standard_folder)
    # standard_list = [case[: case.index('.jpg')] for case in standard_list]
    # target_list = os.listdir(target_folder)
    # target_list = [case[: case.index('_-_slice')] for case in target_list]
    #
    # case_list = [case for case in standard_list if case not in target_list]
    # for case in case_list:
    #     print(case)
    # pass
































