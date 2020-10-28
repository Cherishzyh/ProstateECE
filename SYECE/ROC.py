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
import seaborn as sns

color_patten = sns.color_palette()

from SSHProject.BasicTool.MeDIT.Statistics import BinaryClassification

def Scatter(y_pred, y_true, threshold, *args, **kwargs):
    hy_pred = deepcopy(y_pred)
    hy_pred[y_pred >= threshold] = 1
    hy_pred[y_pred < threshold] = 0
    matrix = confusion_matrix(y_true, hy_pred)
    spe = matrix[0, 0] / np.sum(matrix[0, :])
    sen = matrix[1, 1] / np.sum(matrix[1, :])
    plt.scatter(1 - spe, sen, *args, **kwargs)

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
    # plt.plot(fpn, sen, label='PAGNet: {:.3f}'.format(auc))
    plt.plot(fpn, sen, label='ResNeXt with attention map: {:.3f}'.format(auc))
    fpn, sen, the = roc_curve(label, cnn_w)
    auc = roc_auc_score(label, cnn_w)
    plt.plot(fpn, sen, label='ResNeXt without attention map: {:.3f}'.format(auc))

    # Scatter(hy.values, label, 1/3, s=70, marker='+', label='Reader1 $\geq$ 1', color=color_patten[2])
    # Scatter(hy.values, label, 2/3, s=70, marker='o', label='Reader1 $\geq$ 2', color=color_patten[2])
    # Scatter(bj.values, label, 1/3, s=70, marker='+', label='Reader2 $\geq$ 1', color=color_patten[3])
    # Scatter(bj.values, label, 2/3, s=70, marker='o', label='Reader2 $\geq$ 2', color=color_patten[3])

    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.show()



ROC(r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH_modelvsclinical.csv',
    r'/home/zhangyihong/Documents/ProstateECE/Result/JSPH')
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
