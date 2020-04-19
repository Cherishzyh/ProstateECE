# 分割模型（参考周五段毅汇报的那篇综述，中间有一块描述怎么计算）：
# 模型输入有两个：模型分割的ROI（二值化的np.array 2D/3D）和模型真实ROI（二值化的np.array 2D/3D）。

# 如果是多标签分类，每个做单独统计

import numpy as np
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


#####################################################################

class AreaOrVolumePercentDistance(nn.Module):
    def __init__(self):
        super(AreaOrVolumePercentDistance, self).__init__()

    def forward(self, input, target):
        pass


class MeanSurfaceDistance(nn.Module):
    def __init__(self):
        super(MeanSurfaceDistance, self).__init__()

    def forward(self, input, target):

        if isinstance(input, torch.Tensor):
            input = sitk.GetImageFromArray(input, sitk.sitkUInt8)
        if isinstance(target, torch.Tensor):
            target = sitk.GetImageFromArray(target, sitk.sitkUInt8)

        segmented_surface = sitk.LabelContour(input)

        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(target, squaredDistance=False, useImageSpacing=True))

        label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
        label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)
        surface_distance_results = label_intensity_statistics_filter.GetMean()

        return surface_distance_results


class AverageBoundaryDistance(nn.Module):
    def __init__(self):
        super(AverageBoundaryDistance, self).__init__()

    def forward(self, input, target):
        pass


class AverageRelativeVolumeDifference(nn.Module):
    def __init__(self):
        super(AverageRelativeVolumeDifference, self).__init__()

    def forward(self, input, target):
        pass


#####################################################################

def TestDice(input, target):
    dice = Dice()
    a = dice.forward(input, target)
    print(a)


def TestSensitivityAndSpecificity(input, target):
    ss = SensitivityAndSpecificity()
    TN, TP, FN, FP = ss.Compute(input, target)
    Sensitivity = ss.Sensitivity(input, target)
    Specificity = ss.Specificity(input, target)
    print(Sensitivity, Specificity)


def TestHausdorffDistance(input, target):
    print(type(input))
    HD = HausdorffDistance()
    hd = HD.forward(input, target)
    print(hd)


def TestMeanSurfaceDistance(input, target):
    MSF = MeanSurfaceDistance()
    msf = MSF.forward(input, target)
    print(msf)


if __name__ == '__main__':
    input = torch.from_numpy(np.asarray([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32))
    target = torch.from_numpy(np.asarray([[0, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=np.float32))

    TestMeanSurfaceDistance(input, target)
















