import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, levene, kstest, normaltest, wilcoxon, mannwhitneyu
from scipy.stats import chi2_contingency
from sklearn import metrics
from scipy import stats


def TTestAge(train_age, test_age):
    print("train:", np.mean(train_age), np.std(train_age),
          np.quantile(train_age, 0.25, interpolation='lower'), np.quantile(train_age, 0.75, interpolation='higher'))
    print("test:", np.mean(test_age), np.std(test_age),
          np.quantile(test_age, 0.25, interpolation='lower'), np.quantile(test_age, 0.75, interpolation='higher'))

    # print(normaltest(train_age))
    # print(normaltest(test_age))
    # print(normaltest(train_age + test_age))
    # print(kstest(train_age, 'norm', (np.mean(train_age), np.std(train_age))))
    # print(kstest(test_age, 'norm', (np.mean(test_age), np.std(test_age))))
    # print(levene(train_age, test_age))
    print(mannwhitneyu(train_age, test_age, alternative='two-sided'))
# TTestAge(train_age, test_age)


def CountGrade(Gs_list, grade_list):
    num_list = []
    for grade in grade_list:
        # if grade == grade_list[-1]:
        #     num_list.append(len([case for case in Gs_list if case >= grade]))
        # else:
        num_list.append(len([case for case in Gs_list if case == grade]))
    print(num_list)
    return num_list
# CountbGs(train_pGs, grade_list=[1, 2, 3, 4, 5, 6, 7])
# CountbGs(test_pGs, grade_list=[1, 2, 3, 4, 5, 6, 7])
# CountPIRADS(train_PIRADS, grade_list=[2, 3, 4, 5])
# CountPIRADS(test_PIRADS, grade_list=[2, 3, 4, 5])


def Countpsa(train_psa, test_psa):
    print("train:", np.mean(train_psa), np.std(train_psa),
          np.quantile(train_psa, 0.25, interpolation='lower'), np.quantile(train_psa, 0.75, interpolation='higher'))
    print("test:", np.mean(test_psa), np.std(test_psa),
          np.quantile(test_psa, 0.25, interpolation='lower'), np.quantile(test_psa, 0.75, interpolation='higher'))

    # print(normaltest(train_psa))
    # print(normaltest(test_psa))
    # print(kstest(train_psa, 'norm', (np.mean(train_psa), np.std(train_psa))))
    # print(kstest(test_psa, 'norm', (np.mean(test_psa), np.std(test_psa))))
    # print(levene(train_psa, test_psa))
    print(mannwhitneyu(train_psa, test_psa, alternative='two-sided'))
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


def ConfusionMatrix(prediction, label):
    metric = {}
    prediction, label = np.array(prediction), np.array(label)
    metric['Sample Number'] = len(label)
    metric['Positive Number'] = np.sum(label)
    metric['Negative Number'] = len(label) - np.sum(label)

    fpr, tpr, threshold = metrics.roc_curve(label, prediction)
    index = np.argmax(1 - fpr + tpr)
    metric['Youden Index'] = threshold[index]

    pred = np.zeros_like(label)
    pred[prediction >= threshold[index]] = 1
    C = metrics.confusion_matrix(label, pred, labels=[1, 0])

    metric['accuracy'] = np.where(pred == label)[0].size / label.size
    if np.sum(C[0, :]) < 1e-6:
        metric['sensitivity'] = 0
    else:
        metric['sensitivity'] = C[0, 0] / np.sum(C[0, :])
    if np.sum(C[1, :]) < 1e-6:
        metric['specificity'] = 0
    else:
        metric['specificity'] = C[1, 1] / np.sum(C[1, :])
    if np.sum(C[:, 0]) < 1e-6:
        metric['PPV'] = 0
    else:
        metric['PPV'] = C[0, 0] / np.sum(C[:, 0])
    if np.sum(C[:, 1]) < 1e-6:
        metric['NPV'] = 0
    else:
        metric['NPV'] = C[1, 1] / np.sum(C[:, 1])

    return metric


def ComputeIC(label, prediction):
    bootstrapped_sen, bootstrapped_spe, bootstrapped_PPV, bootstrapped_NPV, = [], [], [], []
    np.random.seed(42)  # control reproducibility
    seed_index = np.random.randint(0, 65535, 1000)
    for seed in seed_index.tolist():
        np.random.seed(seed)
        pred_one_sample = np.random.choice(prediction, size=prediction.size, replace=True)
        np.random.seed(seed)
        label_one_sample = np.random.choice(label, size=label.size, replace=True)

        if len(np.unique(label_one_sample)) < 2:
            continue

        score = ConfusionMatrix(pred_one_sample, label_one_sample)
        bootstrapped_sen.append(score['sensitivity'])
        bootstrapped_spe.append(score['specificity'])
        bootstrapped_PPV.append(score['PPV'])
        bootstrapped_NPV.append(score['NPV'])

    sorted_scores = np.array(bootstrapped_sen)
    # plt.hist(bootstrapped_sen, bins=20)
    # plt.show()
    std_sen, mean_sen = np.std(sorted_scores), np.mean(sorted_scores)
    ci_sen = stats.norm.interval(0.95, loc=mean_sen, scale=std_sen)

    sorted_scores = np.array(bootstrapped_spe)
    # plt.hist(bootstrapped_spe, bins=20)
    # plt.show()
    std_spe, mean_spe = np.std(sorted_scores), np.mean(sorted_scores)
    ci_spe = stats.norm.interval(0.95, loc=mean_spe, scale=std_spe)

    sorted_scores = np.array(bootstrapped_PPV)
    # plt.hist(bootstrapped_PPV, bins=20)
    # plt.show()
    std_PPV, mean_PPV = np.std(sorted_scores), np.mean(sorted_scores)
    ci_PPV = stats.norm.interval(0.95, loc=mean_PPV, scale=std_PPV)

    sorted_scores = np.array(bootstrapped_NPV)
    # plt.hist(bootstrapped_NPV, bins=20)
    # plt.show()
    std_NPV, mean_NPV = np.std(sorted_scores), np.mean(sorted_scores)
    ci_NPV = stats.norm.interval(0.95, loc=mean_NPV, scale=std_NPV)

    print(ci_sen, ci_spe, ci_PPV, ci_NPV)

    return ci_sen, ci_spe, ci_PPV, ci_NPV


def ComputeAllPredP(type='test'):
    if type == 'test':
        pred_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\all_pred_test.csv')
    else:
        pred_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\all_pred_train.csv')
    label = []
    clinical_pred = []
    PAGNet_pred = []
    PAGNet_C_pred = []
    ECE_C_pred = []
    ECE_PAGNet_pred = []
    ECE_C_PAGNet_pred = []
    ECE_pred = []
    for index in pred_df.index:
        label.append(pred_df.loc[index]['Label'])
        clinical_pred.append(pred_df.loc[index]['Clin'])
        PAGNet_pred.append(pred_df.loc[index]['PAGNet'])
        ECE_pred.append(pred_df.loc[index]['MR_ECE'])
        ECE_PAGNet_pred.append(pred_df.loc[index]['PAGNet-MR_ECE'])
        PAGNet_C_pred.append(pred_df.loc[index]['PAGNet-Clin'])
        ECE_C_pred.append(pred_df.loc[index]['MR_ECE-Clin'])
        ECE_C_PAGNet_pred.append(pred_df.loc[index]['PAGNet-MR_ECE-Clin'])

    pred_list = [clinical_pred, ECE_pred, PAGNet_pred, ECE_C_pred, PAGNet_C_pred, ECE_PAGNet_pred, ECE_C_PAGNet_pred]
    pred_list_name = ['clinical_pred', 'ECE_pred', 'PAGNet_pred', 'ECE_C_pred', 'PAGNet_C_pred', 'ECE_PAGNet_pred', 'ECE_C_PAGNet_pred']
    for i, list in enumerate(pred_list):
        for j in range(len(pred_list)):
            if i != j:
                print('{} & {}: {}'.format(pred_list_name[i], pred_list_name[j], wilcoxon(list, pred_list[j])))


def StatisticsClinical(clinical_info, csv_path):
    if isinstance(clinical_info, str):
        clinical_info = [clinical_info]

    df = pd.read_csv(csv_path)

    info_list = []
    all_info_list = []
    for clinical in clinical_info:
        for index in df.index:
            info_list.append(float(df.loc[index][clinical]))
        all_info_list.append(info_list)

    return all_info_list
    # print(mannwhitneyu(train_pz, test_pz))
    # print(mannwhitneyu(train_pGs, test_pGs))
    # print(mannwhitneyu(train_PIRADS, test_PIRADS))
    # print(mannwhitneyu(train_pSVI, test_pSVI))
    # print(mannwhitneyu(train_pSMP, test_pSMP))
    # print(mannwhitneyu(train_psa, test_psa))
    # print(mannwhitneyu(train_pECE, test_pECE))


if __name__ == '__main__':


    # print('excluded_case: {}'.format(excluded_case))
    # print('no_psa_case: {}'.format(no_psa))
    # print('no_LN_case: {}'.format(no_LN))
    # label, ece_score_hy, ece_score_bj = JSPHCSV()
    # ece_pred_bj = [score / 3 for score in ece_score_bj]
    # ece_pred_hy = [score / 3 for score in ece_score_hy]
    # TP_1_bj, TN_1_bj, FP_1_bj, FN_1_bj, sensitivity, specificity, PPV = Compute(label, ece_score_bj, threshould=1)
    # print('TP_1_bj: {}, TN_1_bj: {}, FP_1_bj: {}, FN_1_bj: {}, sensitivity:{}, specificity:{}, PPV:{}'
    #       .format(TP_1_bj, TN_1_bj, FP_1_bj, FN_1_bj, sensitivity, specificity, PPV))
    # TP_1_hy, TN_1_hy, FP_1_hy, FN_1_hy, sensitivity, specificity, PPV = Compute(label, ece_score_hy, threshould=1)
    # print('TP_1_hy: {}, TN_1_hy: {}, FP_1_hy: {}, FN_1_hy: {}, sensitivity:{}, specificity:{}, PPV:{}'
    #       .format(TP_1_hy, TN_1_hy, FP_1_hy, FN_1_hy, sensitivity, specificity, PPV))
    # TP_2_bj, TN_2_bj, FP_2_bj, FN_2_bj, sensitivity, specificity, PPV = Compute(label, ece_score_bj, threshould=2)
    # print('TP_2_bj: {}, TN_2_bj: {}, FP_2_bj: {}, FN_2_bj: {}, sensitivity:{}, specificity:{}, PPV:{}'
    #       .format(TP_2_bj, TN_2_bj, FP_2_bj, FN_2_bj, sensitivity, specificity, PPV))
    # TP_2_hy, TN_2_hy, FP_2_hy, FN_2_hy, sensitivity, specificity, PPV = Compute(label, ece_score_hy, threshould=2)
    # print('TP_2_hy: {}, TN_2_hy: {}, FP_2_hy: {}, FN_2_hy: {}, sensitivity:{}, specificity:{}, PPV:{}'
    #       .format(TP_2_hy, TN_2_hy, FP_2_hy, FN_2_hy, sensitivity, specificity, PPV))
    #
    # print(ComputeAUC(np.array(label), np.array(ece_pred_bj), ci_index=0.95))
    # print(ComputeAUC(np.array(label), np.array(ece_pred_hy), ci_index=0.95))

    # from SYECE.ModelCompared import ModelJSPH, ModelSUH

    # mean_pred, mean_label = ModelJSPH(is_dismap=True, data_type='alltrain')
    # mean_pred_suh, mean_label_suh = ModelSUH(is_dismap=True)
    # mean_pred_nodis, mean_label_nodis = ModelJSPH(is_dismap=False, data_type='alltrain')
    # mean_pred_nodis, mean_label_nodis = ModelSUH(is_dismap=False)
    # np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_TRAIN_dis_result.npy', mean_pred)
    # np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_TRAIN_nodis_result.npy', mean_pred_nodis)
    # np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_hy_result.npy',
    #         np.array(ece_pred_hy))
    # np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_bj_result.npy',
    #         np.array(ece_pred_bj))
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


    # train_csv = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\train_clinical.csv'
    # test_csv = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\test_clinical.csv'
    # train_list = StatisticsClinical('core', train_csv)
    # test_list = StatisticsClinical('core', test_csv)
    # print(mannwhitneyu(train_list[0], test_list[0]))

    # pred_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\ResNeXt\test.csv')
    # pred_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\all_pred_test.csv')
    #
    # label = []
    # pred = []
    # for index in pred_df.index:
    #     label.append(pred_df.loc[index]['Label'])
    #     pred.append(pred_df.loc[index]['MR_ECE'])

    # ComputeAUC(np.array(label), np.array(clinical_pred), ci_index=0.95)
    # ComputeIC(np.array(label), np.array(pred))
    from BasicTool.MeDIT.Statistics import BinaryClassification

    # import numpy as np
    # from scipy.stats import chi2_contingency
    #
    # d = np.array([[265, 66], [331, 84]])
    # print(chi2_contingency(d))
    ece_csv_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ECE-ROI.csv'
    case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case_list = os.listdir(case_folder)
    test_ref = r'X:\FAEFormatData\ECE\test_ref.csv'
    test_ref_df = pd.read_csv(test_ref, index_col='CaseName')

    test_list = test_ref_df.index.tolist()
    train_list = [case for case in case_list if case not in test_list]

    clinical_info = pd.read_csv(ece_csv_path, encoding='gbk', index_col='case')
    age_train_list, age_test_list = [], []
    psa_train_list, psa_test_list = [], []
    PIRADS_train_list, PIRADS_test_list = [], []
    ECE_train_list, ECE_test_list = [], []

    for case in test_list:
        age_train_list.append(int(clinical_info.loc[case]['age']))
        psa_train_list.append(float(clinical_info.loc[case]['psa']))
        PIRADS_train_list.append(int(clinical_info.loc[case]['PI-RADS']))
        ECE_train_list.append(int(clinical_info.loc[case]['pECE']))
    for case in train_list:
        age_test_list.append(int(clinical_info.loc[case]['age']))
        psa_test_list.append(float(clinical_info.loc[case]['psa']))
        PIRADS_test_list.append(int(clinical_info.loc[case]['PI-RADS']))
        ECE_test_list.append(int(clinical_info.loc[case]['pECE']))
    # TTestAge(age_train_list, age_test_list)
    # Countpsa(psa_train_list, psa_test_list)
    CountGrade(PIRADS_train_list, [1, 2, 3, 4, 5])
    CountGrade(PIRADS_test_list, [1, 2, 3, 4, 5])
    print(mannwhitneyu(PIRADS_train_list, PIRADS_test_list, alternative='two-sided'))















