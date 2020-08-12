# 分割模型（参考周五段毅汇报的那篇综述，中间有一块描述怎么计算）：
# 模型输入有两个：模型分割的ROI（二值化的np.array 2D/3D）和模型真实ROI（二值化的np.array 2D/3D）。
# 如果是多标签分类，每个做单独统计

# Relative volume difference (RVD)   1
# symmetric volume difference (SVD)  1
# volumetric overlap error (VOE)  1
# Jaccard similarity coefficient (Jaccard)  1
# Average symmetric surface distance (ASD)  1
# Root mean square symmetric surface distance (RMSD)  1
# Maximum symmetric surface distance (MSD) 1

from enum import Enum
import numpy as np
import SimpleITK as sitk

import torch
import torch.nn as nn

####################################################################
# Done
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
# TODO
class VolumetricOverlapError(nn.Module):
    def __init__(self):
        super(VolumetricOverlapError, self).__init__()

    def forward(self, input, target):
        pass


class JaccardSimilarityCoefficient(nn.Module):
    def __init__(self):
        super(JaccardSimilarityCoefficient, self).__init__()

    def forward(self, input, target):
        pass


class VolumeDifference(nn.Module):
    def __init__(self):
        super(VolumeDifference, self).__init__()

    def forward(self, input, target):
        pass


#####################################################################
# SurfaceDistanceMeasures
class SurfaceDistanceMeasures(Enum):
    hausdorff_distance, max_surface_distance, avg_surface_distance, median_surface_distance, std_surface_distance = range(5)


def ITK(segmentation, reference_segmentation):
    surface_distance_results = np.zeros((1, len(SurfaceDistanceMeasures.__members__.items())))

    segmented_surface = sitk.LabelContour(segmentation)
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False, useImageSpacing=True))

    label_intensity_statistics_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_statistics_filter.Execute(segmented_surface, reference_distance_map)

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.Execute(reference_segmentation, segmentation)

    surface_distance_results[0, SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    surface_distance_results[0, SurfaceDistanceMeasures.max_surface_distance.value] = label_intensity_statistics_filter.GetMaximum(1)
    surface_distance_results[0, SurfaceDistanceMeasures.avg_surface_distance.value] = label_intensity_statistics_filter.GetMean(1)
    surface_distance_results[0, SurfaceDistanceMeasures.median_surface_distance.value] = label_intensity_statistics_filter.GetMedian(1)
    surface_distance_results[0, SurfaceDistanceMeasures.std_surface_distance.value] = label_intensity_statistics_filter.GetStandardDeviation(1)

    # surface_distance_results_df = pd.DataFrame(data=surface_distance_results, index = list(range(1)),
    #                               columns=[name for name, _ in SurfaceDistanceMeasuresITK.__members__.items()])
    return surface_distance_results


def Way2(segmentation, reference_segmentation):
    surface_distance_results = np.zeros((1, len(SurfaceDistanceMeasures.__members__.items())))
    label = 1
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(reference_segmentation, squaredDistance=False))
    reference_surface = sitk.LabelContour(reference_segmentation)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum())
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    hausdorff_distance_filter.Execute(reference_segmentation, segmentation)
    surface_distance_results[
        0, SurfaceDistanceMeasures.hausdorff_distance.value] = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(segmentation, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(segmentation)

    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)

    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))

    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two
    # segmentations, though in our case it is. More on this below.
    surface_distance_results[0, SurfaceDistanceMeasures.avg_surface_distance.value] = np.mean(all_surface_distances)
    surface_distance_results[0, SurfaceDistanceMeasures.median_surface_distance.value] = np.median(
        all_surface_distances)
    surface_distance_results[0, SurfaceDistanceMeasures.std_surface_distance.value] = np.std(all_surface_distances)
    surface_distance_results[0, SurfaceDistanceMeasures.max_surface_distance.value] = np.max(all_surface_distances)

    return surface_distance_results

#####################################################################
# TestDone


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
    # label
    # reference_segmentation = sitk.ReadImage('tumorSegm', sitk.sitkUInt8)
    # prediction
    # segmentation = sitk.ReadImage('tumorSegm2', sitk.sitkUInt8)
    msf0 = ITK(input, target)
    msf2 = Way2(input, target)
    # MSF = MeanSurfaceDistance()
    # msf = MSF.forward(input, target)
    print(msf0)
    print(msf2)



####################################################################

if __name__ == '__main__':
    input = torch.from_numpy(np.asarray([[0, 0, 0], [0, 0, 0], [0, 1, 0]]))
    target = torch.from_numpy(np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))

    input = sitk.GetImageFromArray(input)
    target = sitk.GetImageFromArray(target)
    TestHausdorffDistance(input, target)

    TestMeanSurfaceDistance(input, target)
















