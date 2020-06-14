import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import math

from sklearn import metrics
from imblearn.metrics import specificity_score
from sklearn.calibration import calibration_curve


def get_auc(prob_list, label_list):
    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc

def draw_roc(fpr_list, tpr_list, auc_list, name_list):
    for idx in range(len(fpr_list)):
        label = name_list[idx] + ': ' + '%.3f'%auc_list[idx]
        plt.plot(fpr_list[idx], tpr_list[idx], label=label)

    plt.plot([0, 1], [0, 1], '--', color='r', label='Luck')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def compute_ic(prob_list, label_list, process_viewing=False):
    case_num = len(prob_list)
    auc_list = []
    for i in range(1000):
        if i % 50 == 0 and process_viewing:
            print('*', end=' ')
        df = pd.DataFrame({'prob':prob_list, 'label':label_list})
        df_resample = df.sample(n=case_num, replace=True)
        label_resample = df_resample['label'].values
        prob_resample = df_resample['prob'].values

        auc = get_auc(prob_resample, label_resample, draw=False)
        if math.isnan(auc):
            i = i-1
            continue
        # if not auc:
        #     print(auc)
        #     print('prob',prob_resample, 'label',label_resample)
        auc_list.append(auc)

    auc_list = np.array(auc_list)
    mean, std = auc_list.mean(), auc_list.std(ddof=1)
    conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)
    print('AUC CI: ', conf_intveral)
    return auc_list


# 缺少NPV
def compute_confusion_matrix(pred_list, label_list, model_name='Unnamed'):
    acc = metrics.accuracy_score(label_list, pred_list)
    sen = metrics.recall_score(label_list, pred_list)
    spe = specificity_score(label_list, pred_list)
    ppv = metrics.precision_score(label_list, pred_list)
    youden_index = sen + spe - 1
    print('{}: \tacc: {:.3f},\tsen: {:.3f},\tspe: {:.3f}\n,\tppv:{:.3f},\tyouden index:{:.3f}'.
          format(model_name, acc, sen, spe, ppv, youden_index))


def draw_prob(prob_list, label_list, label_color_dict, case_name=None):
    if not case_name:
        case_name = [str(x) for x in range(len(prob_list))]

    df = pd.DataFrame({'prob': prob_list, 'label': label_list, 'case_name': case_name})
    df = df.sort_values('prob')

    bar_color = [label_color_dict[x] for x in df['label'].values]

    plt.bar(df['case_name'].values, df['prob'].values, color=bar_color)
    plt.show()


def draw_box(prob_list, label_list):
    # df = pd.DataFrame({'prob': prob_list, 'label': label_list})
    category_dict = {}

    for i in range(len(prob_list)):
        label = label_list[i]
        prob = prob_list[i]
        if label in category_dict.keys():
            category_dict[label].append(prob)
        else:
            category_dict[label] = [prob]

    df = pd.DataFrame(category_dict)
    df.plot.box()
    plt.xlabel('Category')
    plt.ylabel('Prob')
    plt.show()


# 未完成
def draw_calibration_curve(prob_list, label_list):
    prod_true, prod_pred = calibration_curve(label_list, prob_list)
    pass


if __name__ == '__main__':
    case_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    prob = [0.1, 0.2, 0.3, 0.5, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    pred = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    label = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]

#    print(get_auc(prob, label))
    # compute_ic(prob, label)
    # compute_confusion_matrix(pred, label)
    # draw_prob(prob, label, {0: 'b', 1: 'r'}, case_name)
    draw_box(prob, label)
