import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, levene, kstest, normaltest, wilcoxon, mannwhitneyu

# def JPSH():
train_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice'
test_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice/Test'
SUH_folder = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice'

ece_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ECE-ROI.csv'


def TTestAge(train_age, test_age):
    print(np.mean(train_age), np.std(train_age),
          np.quantile(train_age, 0.25, interpolation='lower'), np.quantile(train_age, 0.75, interpolation='higher'))
    print(np.mean(test_age), np.std(test_age),
          np.quantile(test_age, 0.25, interpolation='lower'), np.quantile(test_age, 0.75, interpolation='higher'))

    print(normaltest(train_age))
    print(normaltest(test_age))
    print(normaltest(train_age + test_age))
    print(kstest(train_age, 'norm', (np.mean(train_age), np.std(train_age))))
    print(kstest(test_age, 'norm', (np.mean(test_age), np.std(test_age))))
    print(levene(train_age, test_age))
    print(ttest_ind(train_age, test_age))
# TTestAge(train_age, test_age)


def CountGrade(pGs_list, grade_list):
    num_list = []
    for grade in grade_list:
        num_list.append(np.sum([case == grade for case in pGs_list]))
    print(num_list)
    return num_list
# CountbGs(train_pGs, grade_list=[1, 2, 3, 4, 5, 6, 7])
# CountbGs(test_pGs, grade_list=[1, 2, 3, 4, 5, 6, 7])
# CountPIRADS(train_PIRADS, grade_list=[2, 3, 4, 5])
# CountPIRADS(test_PIRADS, grade_list=[2, 3, 4, 5])


def Countpsa(train_psa, test_psa):
    print(np.mean(train_psa), np.std(train_psa),
          np.quantile(train_psa, 0.25, interpolation='lower'), np.quantile(train_psa, 0.75, interpolation='higher'))
    print(np.mean(test_psa), np.std(test_psa),
          np.quantile(test_psa, 0.25, interpolation='lower'), np.quantile(test_psa, 0.75, interpolation='higher'))

    print(normaltest(train_psa))
    print(normaltest(test_psa))
    # print(kstest(train_psa, 'norm', (np.mean(train_psa), np.std(train_psa))))
    # print(kstest(test_psa, 'norm', (np.mean(test_psa), np.std(test_psa))))
    # print(levene(train_psa, test_psa))
    # print(ttest_ind(train_psa, test_psa))
# Countpsa(train_psa, test_psa)


def CountNP(data_list):
    positive = np.sum(data_list)
    negative = len(data_list) - np.sum(data_list)
    print(positive, negative)
    return positive, negative
# CountSMP(train_pSMP)
# CountSMP(test_pSMP)
# CountSVI(train_pSVI)
# CountSVI(test_pSVI)
# CountPZ(train_pz)
# CountPZ(test_pz)
# CountECE(train_pECE)
# CountECE(test_pECE)


def CountECEScore(ece_score):
    num_0 = np.sum([case == 0 for case in ece_score])
    num_1 = np.sum([case == 1 for case in ece_score])
    num_2 = np.sum([case == 2 for case in ece_score])
    num_3 = np.sum([case == 3 for case in ece_score])
    # num_4 = np.sum([case == 4 for case in ece_score])
    print(num_0, num_1, num_2, num_3)
# CountECEScore(ece_score_bj)
# CountECEScore(ece_score_hy)


def SUHCSV(is_tocsv=False):
    SUH_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice'
    ece_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/SUH_ECE_clinical-report.csv'
    case_list = os.listdir(SUH_path)
    case_list = [case[:case.index('_-_')] for case in case_list]

    df = pd.read_csv(ece_path, encoding='gbk', usecols=['case', 'ECE score bj', 'ECE score hy', '包膜突破'],
                     index_col=['case'])

    ece_score_bj, ece_score_hy, label = [], [], []
    col = []
    for index in sorted(df.index):
        if not index in case_list:
            continue
        else:
            info = df.loc[index]
            ece_score_bj.append(int(info['ECE score bj']))
            ece_score_hy.append(int(info['ECE score hy']))
            label.append(int(info['包膜突破']))
            col.append(index)
    # CountECEScore(ece_score_bj)
    # CountECEScore(ece_score_hy)
    if is_tocsv:
        df = pd.DataFrame({'ECE score bj': ece_score_bj, 'ECE score hy': ece_score_hy, 'label': label}, index=col)
        df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/SUH-clinical.csv')
    return label, ece_score_hy, ece_score_bj


def JSPHCSV(is_csv=False):
    # JPSH_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice/Test'
    JPSH_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice'
    ece_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ECE-JSPH-clinical_report.csv'
    case_list = os.listdir(JPSH_path)
    case_list = [case[:case.find('_slice')] for case in case_list]
    case_list.remove('Tes')

    df = pd.read_csv(ece_path, usecols=['case', 'ECE stage bj', 'ECE stage hy', 'pECE'],
                     index_col=['case'])

    ece_score_bj, ece_score_hy, label = [], [], []
    col = []

    for index in sorted(df.index):
        if not index in case_list:
            continue
        elif index == 'DSR^dai shou rong':
            continue
        else:
            info = df.loc[index]
            ece_score_bj.append(int(info['ECE stage bj']))
            ece_score_hy.append(int(info['ECE stage hy']))
            label.append(int(info['pECE']))
            col.append(index)
    if is_csv:
        df = pd.DataFrame({'ECE score bj': ece_score_bj, 'ECE score hy': ece_score_hy, 'label': label}, index=col)
        df.to_csv(r'C:\Users\ZhangYihong\Desktop\JSPH.csv', mode='a+')
    return label, ece_score_hy, ece_score_bj


def Compute(label_list, pred_list, threshould):
    TP, TN, FP, FN = 0, 0, 0, 0
    for index in range(len(label_list)):
        if label_list[index] == 1:
            if pred_list[index] >= threshould:
                TP += 1
            else:
                FN += 1
        elif label_list[index] == 0:
            if pred_list[index] >= threshould:
                FP += 1
            else:
                TN += 1

    # TPR = TP / (TP + FN)
    # FPR = FP / (FP + TN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    PPV = TP / (TP + FP)

    return TP, TN, FP, FN, sensitivity, specificity, PPV
# print(Compute(label, ece_score_bj, threshould=1))
# print(Compute(label, ece_score_hy, threshould=1))
# print(Compute(label, ece_score_bj, threshould=2))
# print(Compute(label, ece_score_hy, threshould=2))


def ComputeAUC(label_list, pred_list, ci_index=0.95):
    from sklearn import metrics
    from scipy import stats

    fpr, tpr, thresholds = metrics.roc_curve(label_list, pred_list)
    auc = metrics.auc(fpr, tpr)

    bootstrapped_scores = []

    np.random.seed(42)  # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(pred_list, size=pred_list.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(label_list, size=label_list.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            continue

        score = metrics.roc_auc_score(label_one_sample, pred_one_sample)
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    std_auc, mean_auc = np.std(sorted_scores), np.mean(sorted_scores)

    ci = stats.norm.interval(ci_index, loc=mean_auc, scale=std_auc)
    return auc, mean_auc, std_auc, ci


if __name__ == '__main__':
    train_list_original = os.listdir(train_folder)
    train_list = [train_case[: train_case.find('_slice')] for train_case in train_list_original]
    train_list.remove('Tes')

    test_list_original = os.listdir(test_folder)
    test_list = [test_case[: test_case.find('_slice')] for test_case in test_list_original]

    df = pd.read_csv(ece_path, encoding='gbk',
                     usecols=['case', 'pECE', 'age', 'psa', 'PZ', 'PI-RADS', 'pGs', 'p-SVI', 'p_SMP', 'p-LN'],
                     index_col=['case'])

    excluded_case, no_psa, no_LN = [], [], []
    train_age, test_age = [], []
    train_psa, test_psa = [], []
    train_pz, test_pz = [], []
    train_PIRADS, test_PIRADS = [], []
    train_pGs, test_pGs = [], []
    train_pSVI, test_pSVI = [], []
    train_pSMP, test_pSMP = [], []
    train_pLN, test_pLN = [], []
    train_pECE, test_pECE = [], []


    for index in df.index:
        info = df.loc[index]
        if index == 'DSR^dai shou rong':
            continue
        if index in train_list:
            if isinstance(info['psa'], str):
                if '＞' in info['psa']:
                    train_psa.append(float(info['psa'][1:]))
                else:
                    train_psa.append(float(info['psa']))
            else:
                no_psa.append(index)

            if isinstance(info['p-LN'], str):
                train_pLN.append(int(info['p-LN']))
            else:
                no_LN.append(index)

            train_age.append(int(info['age']))
            train_pz.append(int(info['PZ']))
            train_PIRADS.append(int(info['PI-RADS']))
            train_pGs.append(int(info['pGs']))
            train_pSVI.append(int(info['p-SVI']))
            train_pSMP.append(int(info['p_SMP']))
            train_pECE.append(int(info['pECE']))

        elif index in test_list:
            if isinstance(info['psa'], str):
                if '＞' in info['psa']:
                    test_psa.append(float(info['psa'][1:]))
                else:
                    test_psa.append(float(info['psa']))
            else:
                no_psa.append(index)

            if isinstance(info['p-LN'], str):
                train_pLN.append(int(info['p-LN']))
            else:
                no_LN.append(index)

            test_age.append(int(info['age']))
            test_pz.append(int(info['PZ']))
            test_PIRADS.append(int(info['PI-RADS']))
            test_pGs.append(int(info['pGs']))
            test_pSVI.append(int(info['p-SVI']))
            test_pSMP.append(int(info['p_SMP']))
            test_pECE.append(int(info['pECE']))

        else:
            excluded_case.append(index)
    # print(mannwhitneyu(train_pz, test_pz))
    # print(mannwhitneyu(train_pGs, test_pGs))
    # print(mannwhitneyu(train_PIRADS, test_PIRADS))
    # print(mannwhitneyu(train_pSVI, test_pSVI))
    # print(mannwhitneyu(train_pSMP, test_pSMP))
    # print(mannwhitneyu(train_psa, test_psa))
    # print(mannwhitneyu(train_pECE, test_pECE))

    # print('excluded_case: {}'.format(excluded_case))
    # print('no_psa_case: {}'.format(no_psa))
    # print('no_LN_case: {}'.format(no_LN))
    label, ece_score_hy, ece_score_bj = JSPHCSV()
    ece_pred_bj = [score / 3 for score in ece_score_bj]
    ece_pred_hy = [score / 3 for score in ece_score_hy]
    TP_1_bj, TN_1_bj, FP_1_bj, FN_1_bj, sensitivity, specificity, PPV = Compute(label, ece_score_bj, threshould=1)
    print('TP_1_bj: {}, TN_1_bj: {}, FP_1_bj: {}, FN_1_bj: {}, sensitivity:{}, specificity:{}, PPV:{}'
          .format(TP_1_bj, TN_1_bj, FP_1_bj, FN_1_bj, sensitivity, specificity, PPV))
    TP_1_hy, TN_1_hy, FP_1_hy, FN_1_hy, sensitivity, specificity, PPV = Compute(label, ece_score_hy, threshould=1)
    print('TP_1_hy: {}, TN_1_hy: {}, FP_1_hy: {}, FN_1_hy: {}, sensitivity:{}, specificity:{}, PPV:{}'
          .format(TP_1_hy, TN_1_hy, FP_1_hy, FN_1_hy, sensitivity, specificity, PPV))
    TP_2_bj, TN_2_bj, FP_2_bj, FN_2_bj, sensitivity, specificity, PPV = Compute(label, ece_score_bj, threshould=2)
    print('TP_2_bj: {}, TN_2_bj: {}, FP_2_bj: {}, FN_2_bj: {}, sensitivity:{}, specificity:{}, PPV:{}'
          .format(TP_2_bj, TN_2_bj, FP_2_bj, FN_2_bj, sensitivity, specificity, PPV))
    TP_2_hy, TN_2_hy, FP_2_hy, FN_2_hy, sensitivity, specificity, PPV = Compute(label, ece_score_hy, threshould=2)
    print('TP_2_hy: {}, TN_2_hy: {}, FP_2_hy: {}, FN_2_hy: {}, sensitivity:{}, specificity:{}, PPV:{}'
          .format(TP_2_hy, TN_2_hy, FP_2_hy, FN_2_hy, sensitivity, specificity, PPV))

    print(ComputeAUC(np.array(label), np.array(ece_pred_bj), ci_index=0.95))
    print(ComputeAUC(np.array(label), np.array(ece_pred_hy), ci_index=0.95))

    from SYECE.ModelCompared import ModelJSPH, ModelSUH

    # mean_pred, mean_label = ModelJSPH(is_dismap=True, data_type='alltrain')
    # mean_pred_suh, mean_label_suh = ModelSUH(is_dismap=True)
    # mean_pred_nodis, mean_label_nodis = ModelJSPH(is_dismap=False, data_type='alltrain')
    # mean_pred_nodis, mean_label_nodis = ModelSUH(is_dismap=False)
    # np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_TRAIN_dis_result.npy', mean_pred)
    # np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_TRAIN_nodis_result.npy', mean_pred_nodis)
    np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_hy_result.npy',
            np.array(ece_pred_hy))
    np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_bj_result.npy',
            np.array(ece_pred_bj))
    # if np.all(label == mean_label):
    #     np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_label.npy', np.array(label))

    # if np.all(label == mean_label):
        # df = pd.DataFrame({'label': label, 'model pred': mean_pred, 'nodis model pred': mean_pred_nodis,
        #                    'hy pred': [score / 3 for score in ece_score_hy],
        #                    'bj pred': [score / 3 for score in ece_score_bj]})
        # df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_modelvsclinical.csv')
    # print(wilcoxon(mean_pred, [score / 3 for score in ece_score_bj_jsph]))
    # print(wilcoxon(mean_pred, [score / 3 for score in ece_score_hy_jsph]))
    # print(wilcoxon(mean_pred_suh, [score / 3 for score in ece_score_bj_suh]))
    # print(wilcoxon(mean_pred_suh, [score / 3 for score in ece_score_hy_suh]))







