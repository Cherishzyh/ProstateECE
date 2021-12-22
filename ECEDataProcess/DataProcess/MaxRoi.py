import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01
from MeDIT.ArrayProcess import ExtractBlock, ExtractPatch

# from FilePath import resample_folder, csv_path, process_folder


def SelectMaxRoiSlice(roi):
    roi_size = []
    for slice in range(roi.shape[-1]):
        roi_size.append(np.sum(roi[..., slice]))

    max_slice = roi_size.index(max(roi_size))

    return max_slice


def GetRoiSize(roi):
    roi_row = []
    roi_column = []
    for row in range(roi.shape[0]):
        roi_row.append(np.sum(roi[row, ...]))
    for column in range(roi.shape[1]):
        roi_column.append(np.sum(roi[..., column]))

    max_row = max(roi_row)
    max_column = max(roi_column)

    # size_list = [max_row, max_channel]
    return max_row, max_column
    # return size_list


def GetRoiCenter(roi):
    roi_row = []
    roi_column = []
    for up in range(roi.shape[0]):
        roi_row.append(np.sum(roi[up, ...]))
    for left in range(roi.shape[1]):
        roi_column.append(np.sum(roi[..., left]))

    max_row = max(roi_row)
    max_column = max(roi_column)
    row_index = roi_row.index(max_row)
    column_index = roi_column.index(max_column)

    left = np.argmax(roi[row_index])
    up = np.argmax(roi[..., column_index])
    center = (int(up + max_column//2), int(left + max_row//2))
    # center = (int(left + max_row // 2), int(up + max_column // 2))
    right = left + max_row
    bottle = up + max_column
    return center, (int(up), int(bottle), int(left), int(right))


def GetRoiCenterBefore(roi):
    roi_row = []
    roi_column = []
    for row in range(roi.shape[0]):
        roi_row.append(np.sum(roi[row, ...]))
    for column in range(roi.shape[1]):
        roi_column.append(np.sum(roi[..., column]))

    max_row = max(roi_row)
    max_column = max(roi_column)
    row_index = roi_row.index(max_row)
    column_index = roi_column.index(max_column)

    column = np.argmax(roi[row_index])
    row = np.argmax(roi[..., column_index])
    center = [int(row + max_row//2), int(column + max_column//2)]
    return center


def KeepLargest(mask):
    new_mask = np.zeros(mask.shape)
    label_im, nb_labels = ndimage.label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    index = np.argmax(max_volume)
    new_mask[label_im == index+1] = 1
    # print(max_volume, np.sum(new_mask))
    return label_im, nb_labels, new_mask


def test(resample_folder):
    case_list = os.listdir(resample_folder)
    row_list = []
    channel_list = []
    for case in case_list:
        case_path = os.path.join(resample_folder, case)
        roi_path = os.path.join(case_path, 'roi.nii')
        if not os.path.exists(roi_path):
            print('% has no roi'.format(case))
            continue

        _, roi, _ = LoadNiiData(roi_path)

        slice = SelectMaxRoiSlice(roi)
        # max_row, max_channel = GetRoiSize(roi[slice, ...])
        size = GetRoiSize(roi[slice, ...])
        if size[0] >= 100 or size[1] >= 100:
            print(case, ':', size)
    #     row_list.append(max_row)
    #     channel_list.append(max_channel)
    #
    # print('max row is:', max(row_list))
    # print('max channel is:', max(channel_list))


def ShowProblemData(resample_folder, case):
    case_path = os.path.join(resample_folder, case)
    t2_path = os.path.join(case_path, 't2.nii')
    roi_path = os.path.join(case_path, 'roi.nii')
    dwi_path = os.path.join(case_path, 'dwi_Reg.nii')
    adc_path = os.path.join(case_path, 'adc_Reg.nii')

    # Load data
    _, t2, _ = LoadNiiData(t2_path, dtype=np.float32)
    _, dwi, _ = LoadNiiData(dwi_path, dtype=np.float32)
    _, adc, _ = LoadNiiData(adc_path, dtype=np.float32)
    _, roi, _ = LoadNiiData(roi_path, dtype=np.uint8)


    t2 = np.transpose(t2, [1, 2, 0])
    dwi = np.transpose(dwi, [1, 2, 0])
    adc = np.transpose(adc, [1, 2, 0])
    _, _, new_roi = KeepLargest(roi)
    new_roi = np.transpose(new_roi, [1, 2, 0])
    Imshow3DArray(Normalize01(t2), roi=Normalize01(new_roi))
    Imshow3DArray(Normalize01(dwi), roi=Normalize01(new_roi))
    Imshow3DArray(Normalize01(adc), roi=Normalize01(new_roi))

    # _, raw_dwi, _ = LoadNiiData(raw_dwi, dtype=np.float32)
    # Imshow3DArray(Normalize01(np.transpose(dwi0, [1,2,0])))
    # Imshow3DArray(Normalize01(np.transpose(raw_dwi[0, ...], [1, 2, 0])))
    # Imshow3DArray(Normalize01(np.transpose(raw_dwi[1, ...], [1, 2, 0])))
    # Imshow3DArray(Normalize01(np.transpose(raw_dwi[2, ...], [1, 2, 0])))


if __name__ == '__main__':
    pass



