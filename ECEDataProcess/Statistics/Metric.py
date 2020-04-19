# 分割模型（参考周五段毅汇报的那篇综述，中间有一块描述怎么计算）：
# 模型输入有两个：模型分割的ROI（二值化的np.array 2D/3D）和模型真实ROI（二值化的np.array 2D/3D）。

# 如果是多标签分类，每个做单独统计

import SimpleITK as sitk

import torch
import torch.nn as nn


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