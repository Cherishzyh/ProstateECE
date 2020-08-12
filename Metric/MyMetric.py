# 分割模型（参考周五段毅汇报的那篇综述，中间有一块描述怎么计算）：
# 模型输入有两个：模型分割的ROI（二值化的np.array 2D/3D）和模型真实ROI（二值化的np.array 2D/3D）。

# 如果是多标签分类，每个做单独统计

import torch
import torch.nn as nn
import os

import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
import SimpleITK as sitk
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

from SSHProject.BasicTool.MeDIT.Decorator import dict_decorator, figure_decorator


def MyWilcoxon(pred1, pred2):
    return stats.wilcoxon(pred1, pred2)[1]

def ComparePred(pred_list, name_list, func=MyWilcoxon):
    if len(pred_list) != len(name_list):
        raise Exception(r"pred_list length doesn't match the name_list length")

    case_num = len(pred_list)
    result_dict = {}
    for i in range(case_num):
        result_list = []
        for j in range(case_num):
            try:
                result = func(pred_list[i], pred_list[j])
            except ValueError:
                result = None
            result_list.append(result)
        result_dict[name_list[i]] = result_list

    result_df = pd.DataFrame(result_dict)
    result_df.index = name_list

    return result_df


def BoostSample(points, n_samples=1000):

    if isinstance(points, list):
        point_array = np.array(points)
    elif isinstance(points, pd.DataFrame):
        point_array = np.array(points)
    elif isinstance(points, pd.Series):
        point_array = points.values
    elif isinstance(points, np.ndarray):
        point_array = points
    else:
        print('The type of points is : ', type(points))

    samples = []
    for index in range(n_samples):
        one_sample = np.random.choice(point_array, size=point_array.size, replace=True)
        samples.append(one_sample.mean())

    return sorted(samples)


def GetThreshold(prediction, label, own_threshold=None):
    prediction, label = np.array(prediction), np.array(label)

    if len(prediction.shape) == 2:
        prediction = np.squeeze(prediction)
    if len(label.shape) == 2:
        label = np.squeeze(label)

    assert (len(prediction.shape) == 1)
    assert (len(label.shape) == 1)

    fpr, tpr, threshold = metrics.roc_curve(label, prediction)
    index = np.argmax(1 - fpr + tpr)
    metric = threshold[index]

    if own_threshold == None:
        pred = np.zeros_like(label)
        pred[prediction >= threshold[index]] = 1
        C = metrics.confusion_matrix(label, pred, labels=[1, 0])
    else:
        pred = np.zeros_like(label)
        pred[prediction >= own_threshold] = 1
        C = metrics.confusion_matrix(label, pred, labels=[1, 0])
    return metric, C, pred


class BinaryClassification(object):
    def __init__(self, is_show=True, store_folder='', store_format='jpg'):
        self.color_list = sns.color_palette('deep')
        self._metric = {}
        self.UpdateShow(is_show)
        if os.path.isdir(store_folder):
            self.UpdateStorePath(store_folder, store_format)

    def UpdateShow(self, show):
        self._ConfusionMatrix.set_show(show)

        self._DrawRoc.set_show(show)
        self._DrawBox.set_show(show)
        self._DrawProbability.set_show(show)
        self._CalibrationCurve.set_show(show)

    def UpdateStorePath(self, store_folder, store_format='jpg'):
        self._ConfusionMatrix.set_store_path(os.path.join(store_folder, 'ConfusionMatrixInfo.csv'))

        self._DrawRoc.set_store_path(os.path.join(store_folder, 'ROC.{}'.format(store_format)))
        self._DrawBox.set_store_path(os.path.join(store_folder, 'Box.{}'.format(store_format)))
        self._DrawProbability.set_store_path(os.path.join(store_folder, 'Probability.{}'.format(store_format)))
        self._CalibrationCurve.set_store_path(os.path.join(store_folder, 'Calibration.{}'.format(store_format)))

    def __Auc(self, y_true, y_pred, ci_index=0.95):
        """
        This function can help calculate the AUC value and the confidence intervals. It is note the confidence interval is
        not calculated by the standard deviation. The auc is calculated by sklearn and the auc of the group are bootstraped
        1000 times. the confidence interval are extracted from the bootstrap result.

        Ref: https://onlinelibrary.wiley.com/doi/abs/10.1002/%28SICI%291097-0258%2820000515%2919%3A9%3C1141%3A%3AAID-SIM479%3E3.0.CO%3B2-F
        :param y_true: The label, dim should be 1.
        :param y_pred: The prediction, dim should be 1
        :param CI_index: The range of confidence interval. Default is 95%
        """

        single_auc = metrics.roc_auc_score(y_true, y_pred)

        bootstrapped_scores = []

        np.random.seed(42)  # control reproducibility
        seed_index = np.random.randint(0, 65535, 1000)
        for seed in seed_index.tolist():
            np.random.seed(seed)
            pred_one_sample = np.random.choice(y_pred, size=y_pred.size, replace=True)
            np.random.seed(seed)
            label_one_sample = np.random.choice(y_true, size=y_pred.size, replace=True)

            if len(np.unique(label_one_sample)) < 2:
                continue

            score = metrics.roc_auc_score(label_one_sample, pred_one_sample)
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        std_auc, mean_auc = np.std(sorted_scores), np.mean(sorted_scores)

        ci = stats.norm.interval(ci_index, loc=mean_auc, scale=std_auc)
        return single_auc, mean_auc, std_auc, ci

    @dict_decorator()
    def _ConfusionMatrix(self, prediction, label, label_legend, own_threshold=None):
        prediction, label = np.array(prediction), np.array(label)

        if len(prediction.shape) == 2:
            prediction = np.squeeze(prediction)
        if len(label.shape) == 2:
            label = np.squeeze(label)

        assert (len(prediction.shape) == 1)
        assert (len(label.shape) == 1)

        self._metric['Sample Number'] = len(label)
        self._metric['Positive Number'] = np.sum(label)
        self._metric['Negative Number'] = len(label) - np.sum(label)

        fpr, tpr, threshold = metrics.roc_curve(label, prediction)
        index = np.argmax(1 - fpr + tpr)
        self._metric['Youden Index'] = threshold[index]

        if own_threshold == None:
            pred = np.zeros_like(label)
            pred[prediction >= threshold[index]] = 1
            C = metrics.confusion_matrix(label, pred, labels=[1, 0])
        else:
            pred = np.zeros_like(label)
            pred[prediction >= own_threshold] = 1
            C = metrics.confusion_matrix(label, pred, labels=[1, 0])

        self._metric['accuracy'] = np.where(pred == label)[0].size / label.size
        if np.sum(C[0, :]) < 1e-6:
            self._metric['sensitivity'] = 0
        else:
            self._metric['sensitivity'] = C[0, 0] / np.sum(C[0, :])
        if np.sum(C[1, :]) < 1e-6:
            self._metric['specificity'] = 0
        else:
            self._metric['specificity'] = C[1, 1] / np.sum(C[1, :])
        if np.sum(C[:, 0]) < 1e-6:
            self._metric['PPV'] = 0
        else:
            self._metric['PPV'] = C[0, 0] / np.sum(C[:, 0])
        if np.sum(C[:, 1]) < 1e-6:
            self._metric['NPV'] = 0
        else:
            self._metric['NPV'] = C[1, 1] / np.sum(C[:, 1])

        single_auc, mean_auc, std, ci = self.__Auc(label, prediction, ci_index=0.95)
        self._metric['AUC'] = single_auc
        self._metric['95 CIs Lower'], self._metric['95 CIs Upper'] = ci[0], ci[1]
        self._metric['AUC std'] = std

    @figure_decorator()
    def _DrawProbability(self, prediction, label, youden_index=0.5):
        df = pd.DataFrame({'prob': prediction, 'label': label})
        df = df.sort_values('prob')

        bar_color = [self.color_list[x] for x in df['label'].values]
        plt.bar(range(len(prediction)), df['prob'].values - youden_index, color=bar_color)
        plt.yticks([df['prob'].values.min() - youden_index,
                    youden_index - youden_index,
                    df['prob'].max() - youden_index],
                   ['{:.2f}'.format(df['prob'].values.min()),
                    '{:.2f}'.format(youden_index),
                    '{:.2f}'.format(df['prob'].max())
                    ])

    @figure_decorator()
    def _DrawBox(self, prediction, label, label_legend):
        prediction, label = np.array(prediction), np.array(label)
        positive = prediction[label == 1]
        negative = prediction[label == 0]

        sns.boxplot(data=[negative, positive])
        plt.xticks([0, 1], label_legend)

    @figure_decorator()
    def _DrawRoc(self, prediction, label):
        fpr, tpr, threshold = metrics.roc_curve(label, prediction)
        auc = metrics.roc_auc_score(label, prediction)
        name = 'AUC = {:.3f}'.format(auc)

        plt.plot(fpr, tpr, color=self.color_list[0], label=name, linewidth=3)

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

    @figure_decorator()
    def _CalibrationCurve(self, prediction, label):
        F, threshold = calibration_curve(label, prediction, n_bins=10)
        clf_score = metrics.brier_score_loss(label, prediction, pos_label=1)
        plt.plot(threshold, F, "s-", label='{:.3f}'.format(clf_score))
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.ylabel("Fraction of positives")
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right")

    def Run(self, pred, label, label_legend=('Negative', 'Positive'), store_folder=r''):
        assert(isinstance(pred, list))
        assert(isinstance(label, list))

        self._ConfusionMatrix(pred, label, label_legend)
        self._DrawRoc(pred, label)
        self._DrawProbability(pred, label, youden_index=self._metric['Youden Index'])
        self._DrawBox(pred, label, label_legend)
        self._CalibrationCurve(pred, label)

        return self._metric


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):
        smooth = 1

        input_flat = input.view(-1)
        target_flat = target.view(-1)

        intersection = (input_flat * target_flat).sum()
        return (2 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)


class SensitivityAndSpecificity(nn.Module):
    def __init__(self):
        super(SensitivityAndSpecificity, self).__init__()
        self.TN = 0
        self.TP = 0
        self.FN = 0
        self.FP = 0

    def Compute(self, predict, target):

        input_flat = predict.view(-1)
        target_flat = target.view(-1)

        for index in range(len(input_flat)):
            if input_flat[index] == target_flat[index] == 0:
                self.TN += 1
            elif input_flat[index] == target_flat[index] == 1:
                self.TP += 1
            elif input_flat[index] == 1 and target_flat[index] == 0:
                self.FP += 1
            elif input_flat[index] == 0 and target_flat[index] == 1:
                self.FN += 1
        return self.TN, self.TP, self.FN, self.FP

    def Specificity(self, predict, target):
        self.Compute(predict, target)
        specificity = self.TN / (self.TN + self.FP)
        return specificity

    def Sensitivity(self, predict, target):
        self.Compute(predict, target)
        sensitivity = self.TP / (self.TP + self.FN)
        return sensitivity


class HausdorffDistance(nn.Module):
    def __init__(self):
        super(HausdorffDistance, self).__init__()

    def forward(self, input, target):

        if isinstance(input, torch.Tensor):
            input = sitk.GetImageFromArray(input)
        if isinstance(target, torch.Tensor):
            target = sitk.GetImageFromArray(target)

        hausdorff_computer = sitk.HausdorffDistanceImageFilter()
        hausdorff_computer.Execute(input, target)

        return hausdorff_computer.GetHausdorffDistance()