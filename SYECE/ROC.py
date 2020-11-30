"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2020/9/27
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl
import seaborn as sns
from pylab import *


color_patten = sns.color_palette()

from BasicTool.MeDIT.Statistics import BinaryClassification


# def Scatter(y_pred, y_true, threshold, *args, **kwargs):
#     hy_pred = deepcopy(y_pred)
#     hy_pred[y_pred >= threshold] = 1
#     hy_pred[y_pred < threshold] = 0
#     matrix = confusion_matrix(y_true, hy_pred)
#     spe = matrix[0, 0] / np.sum(matrix[0, :])
#     sen = matrix[1, 1] / np.sum(matrix[1, :])
#     plt.scatter(1 - spe, sen, *args, **kwargs)


def ROC(file_path, save_path):
    df = pd.read_csv(file_path, index_col=0)
    print(df.columns)
    label = df['label'].values.astype(int)
    cnn_a, cnn_w = df['model pred'].values, df['nodis model pred'].values
    hy, bj = df['hy pred'], df['bj pred']

    # bc = BinaryClassification()
    # bc.Run(cnn_a.tolist(), label.tolist())

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, the = roc_curve(label, cnn_a)
    auc = roc_auc_score(label, cnn_a)
    plt.plot(fpn, sen, label='PAGNet: {:.3f}'.format(auc))
    plt.plot(fpn, sen, label='ResNeXt with attention map: {:.3f}'.format(auc))
    fpn, sen, the = roc_curve(label, cnn_w)
    auc = roc_auc_score(label, cnn_w)
    plt.plot(fpn, sen, label='ResNeXt without attention map: {:.3f}'.format(auc))

    Scatter(hy.values, label, 1/3, s=70, marker='+', label='Reader1 $\geq$ 1', color=color_patten[2])
    Scatter(hy.values, label, 2/3, s=70, marker='o', label='Reader1 $\geq$ 2', color=color_patten[2])
    Scatter(bj.values, label, 1/3, s=70, marker='+', label='Reader2 $\geq$ 1', color=color_patten[3])
    Scatter(bj.values, label, 2/3, s=70, marker='o', label='Reader2 $\geq$ 2', color=color_patten[3])

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()

# ROC(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_modelvsclinical.csv',
#     r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH')
# ROC(r'/home/zhangyihong/Documents/ProstateECE/Result/SUH_modelvsclinical.csv',
#     r'/home/zhangyihong/Documents/ProstateECE/Result/SUH')
# ROC(r'C:\Users\yangs\Desktop\SUH.csv')


def ROC2(cnn_path, clinical_path, save_path=''):
    cnn_df = pd.read_csv(cnn_path, index_col=0)
    print(cnn_df.columns)
    cnn_label = cnn_df['label'].values.astype(int)
    cnn_a, cnn_w = cnn_df['attention'].values, cnn_df['without attention'].values

    clinical_df = pd.read_csv(clinical_path, index_col=0)
    print(clinical_df.columns)
    clinical_label = clinical_df['label'].values.astype(int)
    hy, bj = clinical_df['ECE score hy'], clinical_df['ECE score bj']

    # bc = BinaryClassification()
    # bc.Run(cnn_a.tolist(), label.tolist())

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, the = roc_curve(cnn_label, cnn_a)
    auc = roc_auc_score(cnn_label, cnn_a)
    plt.plot(fpn, sen, label='CNN with Attention: {:.3f}'.format(auc))
    fpn, sen, the = roc_curve(cnn_label, cnn_w)
    auc = roc_auc_score(cnn_label, cnn_w)
    plt.plot(fpn, sen, label='CNN without Attention: {:.3f}'.format(auc))

    Scatter(hy.values, clinical_label, 0.5, s=70, marker='o', label='Reader1 $\geq$ 1', color=color_patten[2])
    Scatter(hy.values, clinical_label, 1.5, s=70, marker='+', label='Reader1 $\geq$ 2', color=color_patten[2])
    Scatter(bj.values, clinical_label, 0.5, s=70, marker='o', label='Reader2 $\geq$ 1', color=color_patten[3])
    Scatter(bj.values, clinical_label, 1.5, s=70, marker='+', label='Reader2 $\geq$ 2', color=color_patten[3])

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend()

    plt.savefig(save_path + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.show()
    plt.close()

# ROC2(r'/home/zhangyihong/Documents/ProstateECE/cnn_JSPH.csv',
#      r'/home/zhangyihong/Documents/ProstateECE/clinical-JSPH.csv',
#      r'/home/zhangyihong/Documents/ProstateECE/JSPH_ROC')
# ROC2(r'/home/zhangyihong/Documents/ProstateECE/cnn_SUH.csv',
#      r'/home/zhangyihong/Documents/ProstateECE/clinical-SUH.csv',
#      r'/home/zhangyihong/Documents/ProstateECE/SUH_ROC')


def ROCNPY(label, Res_pred, PAG_pred, save_path):
    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, the = roc_curve(label, PAG_pred)
    auc = roc_auc_score(label, PAG_pred)
    plt.plot(fpn, sen, label='PAGNet: {:.3f}'.format(auc))

    fpn, sen, the = roc_curve(label, Res_pred)
    auc = roc_auc_score(label, Res_pred)
    plt.plot(fpn, sen, label='ResNeXt: {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')

    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(save_path + '.jpg', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()



# csv  resnext vs pagnet


def ROC3():
    PAGNet_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\train_right.csv')
    pag_label, pag_pred = [], []
    for index in PAGNet_df.index:
        pag_label.append(PAGNet_df.loc[index]['Label'])
        pag_pred.append(PAGNet_df.loc[index]['PAGNet'])

    ResNeXt_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\ResNeXt\train.csv')
    res_label, res_pred = [], []
    for index in ResNeXt_df.index:
        res_label.append(ResNeXt_df.loc[index]['Label'])
        res_pred.append(ResNeXt_df.loc[index]['PAGNet'])

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, _ = roc_curve(pag_label, pag_pred)
    pag_auc = roc_auc_score(pag_label, pag_pred)
    plt.plot(fpn, sen, label='PAGNet: {:.3f}'.format(pag_auc))

    fpn, sen, _ = roc_curve(res_label, res_pred)
    res_auc = roc_auc_score(res_label, res_pred)
    plt.plot(fpn, sen, label='ResNeXt: {:.3f}'.format(res_auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.savefig(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\train_roc' + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()

# ROC3()

def Metric():
    PAGNet_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\test.csv')
    pag_label, pag_pred = [], []
    for index in PAGNet_df.index:
        pag_label.append(PAGNet_df.loc[index]['Label'])
        pag_pred.append(PAGNet_df.loc[index]['PAGNet'])

    ResNeXt_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\ResNeXt\test.csv')
    res_label, res_pred = [], []
    for index in ResNeXt_df.index:
        res_label.append(ResNeXt_df.loc[index]['Label'])
        res_pred.append(ResNeXt_df.loc[index]['PAGNet'])

    bc = BinaryClassification(store_folder=r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\ResNeXt', store_format='eps')
    bc.Run(res_pred, res_label)


def Scatter(y_pred, y_true, *args, **kwargs):
    hy_pred = deepcopy(y_pred)
    matrix = confusion_matrix(y_true, hy_pred)
    spe = matrix[0, 0] / np.sum(matrix[0, :])
    sen = matrix[1, 1] / np.sum(matrix[1, :])
    plt.scatter(1 - spe, sen, *args, **kwargs)

# Clinical v.s. PAGNet
def ROC4():
    # PAGNet
    pred_df = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\all_pred_test.csv')
    label = []
    clin_pred = []
    pag_pred = []
    PAGNet_C_pred = []
    ECE_C_pred = []
    ECE_PAGNet_pred = []
    ECE_C_PAGNet_pred = []
    ECE_pred = []
    for index in pred_df.index:
        label.append(pred_df.loc[index]['Label'])
        clin_pred.append(pred_df.loc[index]['Clin'])
        pag_pred.append(pred_df.loc[index]['PAGNet'])
        ECE_pred.append(pred_df.loc[index]['MR_ECE'])
        ECE_PAGNet_pred.append(pred_df.loc[index]['PAGNet-MR_ECE'])
        PAGNet_C_pred.append(pred_df.loc[index]['PAGNet-Clin'])
        ECE_C_pred.append(pred_df.loc[index]['MR_ECE-Clin'])
        ECE_C_PAGNet_pred.append(pred_df.loc[index]['PAGNet-MR_ECE-Clin'])

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    Scatter(ECE_pred, label, s=70, marker='o', label='MR-ECE', color=color_patten[6])

    fpn, sen, _ = roc_curve(label, clin_pred)
    ECE_C_auc = roc_auc_score(label, clin_pred)
    plt.plot(fpn, sen, label='Clinical: {:.3f}'.format(ECE_C_auc))

    fpn, sen, _ = roc_curve(label, pag_pred)
    pag_auc = roc_auc_score(label, pag_pred)
    plt.plot(fpn, sen, label='PAGNet: {:.3f}'.format(pag_auc))

    fpn, sen, _ = roc_curve(label, ECE_C_pred)
    ECE_C_auc = roc_auc_score(label, ECE_C_pred)
    plt.plot(fpn, sen, label='MR-ECE + Clinical: {:.3f}'.format(ECE_C_auc))

    fpn, sen, _ = roc_curve(label, PAGNet_C_pred)
    PAGNet_C_auc = roc_auc_score(label, PAGNet_C_pred)
    plt.plot(fpn, sen, label='PAGNet + Clinical: {:.3f}'.format(PAGNet_C_auc))

    fpn, sen, _ = roc_curve(label, ECE_PAGNet_pred)
    PAGNet_C_auc = roc_auc_score(label, ECE_PAGNet_pred)
    plt.plot(fpn, sen, label='PAGNet + MR-ECE: {:.3f}'.format(PAGNet_C_auc))

    fpn, sen, _ = roc_curve(label, ECE_C_PAGNet_pred)
    ECE_C_PAGNet_auc = roc_auc_score(label, ECE_C_PAGNet_pred)
    plt.plot(fpn, sen, label='PAGNet + MR-ECE + Clinical: {:.3f}'.format(ECE_C_PAGNet_auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.savefig(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\CaseBased_NoSigmoid\ROC_7' + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()
# ROC4()

def Test():
    ECE = pd.read_csv(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\CaseBased_NoSigmoid\feature\clinical+ECE_test.csv')
    ECE_label, ECE_pred = [], []
    for index in ECE.index:
        ECE_label.append(ECE.loc[index]['label'])
        ECE_pred.append(ECE.loc[index]['MR_ECE'])
    TP, TN, FP, FN = 0, 0, 0, 0
    for index in range(len(ECE_pred)):
        if ECE_label[index] == 1:
            if ECE_pred[index] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if ECE_pred[index] == 0:
                TN += 1
            else:
                FP += 1
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    print(sen, spe)
    # plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')
    Scatter(ECE_label, ECE_pred, s=70, marker='o', label='MR-ECE', color=color_patten[5])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.savefig(r'C:\Users\ZhangYihong\Desktop\JMRI\feature\CaseBased_NoSigmoid\ROC_4' + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()
# Test()
# matrix = confusion_matrix(ECE_label, ECE_pred)
# print(matrix)
# spe = matrix[0, 0] / np.sum(matrix[0, :])
# sen = matrix[1, 1] / np.sum(matrix[1, :])
# print(sen, spe)
# from BasicTool.MeDIT.Statistics import BinaryClassification
# bc = BinaryClassification()
# bc.Run(ECE_pred, ECE_label)

# pred_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh.csv')
# label = []
# clin_pred = []
# for index in pred_df.index:
#     label.append(pred_df.loc[index]['Label'])
#     clin_pred.append(pred_df.loc[index]['PAGNet'])
# ECE_C_auc = roc_auc_score(label, clin_pred)
# print(ECE_C_auc)

if __name__  == '__main__':
    from SSHProject.BasicTool.MeDIT.Statistics import BinaryClassification
    bc = BinaryClassification()
    label = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/PCaRoi_SUH_label.npy').astype(int).tolist()
    Boundary = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/Boundary_SUH_pred.npy').tolist()
    PCaROI = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/PCaRoi_SUH_pred.npy').tolist()
    Binary = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/BinaryAttentionMap_SUH_pred.npy').tolist()
    Attention = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_dis_result.npy').tolist()
    ResNeXt = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_nodis_result.npy').tolist()

    # label = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/PCaRoi_alltrain_label.npy').astype(int).tolist()
    # Boundary = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/Boundary_alltrain_pred.npy').tolist()
    # PCaROI = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/PCaRoi_alltrain_pred.npy').tolist()
    # Binary = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/ChangeAtten/BinaryAttentionMap_alltrain_pred.npy').tolist()
    # Attention = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_dis_result.npy').tolist()
    # ResNeXt = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_nodis_result.npy').tolist()
    # bc.Run(PCaROI, label)
    # bc.Run(Boundary, label)

    plt.figure(0, figsize=(6, 5))
    plt.plot([0, 1], [0, 1], 'k--')

    fpn, sen, _ = roc_curve(label, ResNeXt)
    auc = roc_auc_score(label, ResNeXt)
    plt.plot(fpn, sen, label='Baseline: {:.3f}'.format(auc))

    fpn, sen, _ = roc_curve(label, Boundary)
    auc = roc_auc_score(label, Boundary)
    plt.plot(fpn, sen, label='Prostate Boundary: {:.3f}'.format(auc))

    fpn, sen, _ = roc_curve(label, PCaROI)
    auc = roc_auc_score(label, PCaROI)
    plt.plot(fpn, sen, label='PCa Roi: {:.3f}'.format(auc))

    fpn, sen, _ = roc_curve(label, Binary)
    auc = roc_auc_score(label, Binary)
    plt.plot(fpn, sen, label='Binary Attention Map: {:.3f}'.format(auc))

    fpn, sen, _ = roc_curve(label, Attention)
    auc = roc_auc_score(label, Attention)
    plt.plot(fpn, sen, label='Attention Map: {:.3f}'.format(auc))

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')

    plt.show()
