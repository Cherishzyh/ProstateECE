import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Normalize import Normalize01
from MeDIT.Visualization import Imshow3DArray

from FilePath import csv_path, resample_folder, cnn_folder, desktop_path, input_0_output_0_path

info = pd.read_csv(csv_path, usecols=['case', 'pECE'], index_col=['case'])


def CropT2Data(t2_data, crop_shape, slice_index, center):
    from MeDIT.ArrayProcess import ExtractPatch
    t2_one_slice = t2_data[slice_index, ...]

    # Normalization
    t2_one_slice -= np.mean(t2_one_slice)
    t2_one_slice /= np.std(t2_one_slice)

    # Crop
    t2_crop, _ = ExtractPatch(t2_one_slice, crop_shape[1:], center_point=center)
    return t2_crop


def CropRoiData(roi_data, crop_shape, slice_index, center):
    from MeDIT.ArrayProcess import ExtractPatch
    roi_one_slice_onehot = roi_data[slice_index, ...]

    # Crop
    roi_crop, _ = ExtractPatch(roi_one_slice_onehot, crop_shape[1:], center_point=center)
    return roi_crop


# MakeH5()
def WriteH5(data_folder, save_path):
    from DataProcess.MaxRoi import SelectMaxRoiSlice, GetRoiCenter, KeepLargest
    case_list = os.listdir(data_folder)
    crop_shape = (1, 280, 280)

    for case in case_list:
        # path
        case = 'XZP^xiang zheng ping'
        case_path = os.path.join(data_folder, case)
        t2_path = os.path.join(case_path, 't2.nii')
        roi_path = os.path.join(case_path, 'roi.nii')
        dwi_path = os.path.join(case_path, 'dwi_Reg.nii')
        adc_path = os.path.join(case_path, 'adc_Reg.nii')
        ece = info.loc[case, 'pECE']

        if ece == 0:
            ece = np.array([0, 1], dtype=np.uint8)
        elif ece == 1:
            ece = np.array([1, 0], dtype=np.uint8)

        # Load data
        _, t2, _ = LoadNiiData(t2_path, dtype=np.float32)
        _, dwi, _ = LoadNiiData(dwi_path, dtype=np.float32)
        _, adc, _ = LoadNiiData(adc_path, dtype=np.float32)
        _, roi, _ = LoadNiiData(roi_path, dtype=np.uint8)

        _, _, new_roi = KeepLargest(roi)

        slice = SelectMaxRoiSlice(new_roi)

        center = GetRoiCenter(new_roi[slice, ...])

        t2_slice = CropT2Data(t2, crop_shape, slice, center=center)
        dwi_slice = CropT2Data(dwi, crop_shape, slice, center=center)
        adc_slice = CropT2Data(adc, crop_shape, slice, center=center)
        roi_slice = CropRoiData(new_roi, crop_shape, slice, center=center)

        t2_slice_3d = t2_slice[np.newaxis, ...]
        dwi_slice_3d = dwi_slice[np.newaxis, ...]
        adc_slice_3d = adc_slice[np.newaxis, ...]
        roi_slice_3d = roi_slice[np.newaxis, ...]
        # store_path = r'C:\Users\ZhangYihong\Desktop\try\image'

        dataname = case + '_slice' + str(slice) + '.h5'
        datapath = os.path.join(save_path, dataname)

        with h5py.File(datapath, 'w') as f:
            f['input_0'] = t2_slice_3d
            f['input_1'] = dwi_slice_3d
            f['input_2'] = adc_slice_3d
            f['output_0'] = roi_slice_3d
            f['output_1'] = ece

        break


def TestWhiteH5():
    number = 0
    case_path = r'X:\StoreFormatData\ProstateCancerECE\ResampleData\XZP^xiang zheng ping'
    t2_path = os.path.join(case_path, 't2.nii')
    roi_path = os.path.join(case_path, 'roi.nii')
    dwi_path = os.path.join(case_path, 'dwi_Reg.nii')
    adc_path = os.path.join(case_path, 'adc_Reg.nii')
    ece = info.loc['LU CHANG JIANG', 'pECE']
    print(ece)
    if ece == 'nan':
        print('nan')
    else:
        # Load data
        _, _, t2 = LoadNiiData(t2_path, dtype=np.float32)
        _, _, dwi = LoadNiiData(dwi_path, dtype=np.float32)
        _, _, adc = LoadNiiData(adc_path, dtype=np.float32)
        _, _, roi = LoadNiiData(roi_path, dtype=np.uint8)

        for index in range(t2.shape[-1]):
            number += 1
            t2_slice = t2[..., index]
            roi_slice = roi[..., index]
            dwi_slice = dwi[..., index]
            adc_slice = adc[..., index]

            dataname = 'data' + str(number) + '.h5'
            datapath = os.path.join(desktop_path, dataname)

            with h5py.File(datapath, 'w') as f:
                # f = h5py.File(datapath, 'w')
                f['input_t2'] = t2_slice
                f['input_dwi'] = dwi_slice
                f['input_adc'] = adc_slice
                f['output_roi'] = roi_slice
                f['output_ece'] = ece


def ShowH5():
    # data read
    path = r'X:\StoreFormatData\ProstateCancerECE\H5Data'
    case_list = os.listdir(path)
    for case in case_list:
        file_path = os.path.join(path, case)
        with h5py.File(file_path, 'r') as h5_file:
            # t2 = np.asarray(h5_file['input_t2'], dtype=np.float32)
            # dwi = np.asarray(h5_file['input_dwi'], dtype=np.float32)
            # adc = np.asarray(h5_file['input_adc'], dtype=np.float32)
            # roi = np.asarray(h5_file['output_roi'], dtype=np.uint8)
            ece = np.asarray(h5_file['output_ece'], dtype=np.uint8)

        if ece == 'nan':
            print(case)

    # return t2, dwi, adc, roi, ece


def Checkb():
    from DataProcess.SelectDWI import NearTrueB
    case_list = ['CSF^chen song fu', 'CYX^chen yu xiang', 'DRJ^dai ru jiang', 'DSB^dai song bo ^^6698-7', 'GJD^guo jin dong',
                 'GU SI KANG', 'HGH^he gong huang', 'HGH^hu guo hua', 'JIANG HONG GEN', 'JLS^jiang li shan', 'LEC^liu er chang ^^6698-13',
                 'LHP^lu hao pei ^^6698-40', 'LJJ^lu qi jia', 'LJY^liu jia yan', 'LU JI SHUN', 'LYZ^liu yin zhong','LZW^li zhong wei',
                 'SCF^shi can fa', 'SXC^su xue cao', 'SXQ^shi xiao quan', 'WR^wu ruo', 'XJH^xue jian hua', 'YU TU JUN', 'YYQ^yang yu qing ^^6698-+5']

    b_value = ['0', '50', '700', '750', '1400', '1500']

    path = r'X:\PrcoessedData\ProstateCancerECE'

    for case in case_list:
        case_path = os.path.join(path, case)
        dwi_bval_path = os.path.join(case_path, 'dwi.bval')
        dki_bval_path = os.path.join(case_path, 'dki.bval')
        if os.path.exists(dwi_bval_path):
            bval = open(dwi_bval_path, 'r')
            b_list = bval.read().split()
            b, index = NearTrueB(b_list)
            _, _, dwi = LoadNiiData(os.path.join(case_path, 'dwi.nii'))
            Imshow3DArray(Normalize01(dwi[..., index]))
            print(case, b)
        elif os.path.exists(dki_bval_path):
            bval = open(dki_bval_path, 'r')
            b_list = bval.read().split()
            b, index = NearTrueB(b_list)
            _, _, dki = LoadNiiData(os.path.join(case_path, 'dki.nii'))
            Imshow3DArray(Normalize01(dki[..., index]))
            print(case, b)
        else:
            for b in reversed(b_value):
                if os.path.exists(os.path.join(case_path, 'dwi_b'+str(b)+'.nii')):
                    _, _, dwi = LoadNiiData(os.path.join(case_path, 'dwi_b'+str(b)+'.nii'))
                    Imshow3DArray(Normalize01(dwi))
                    print(case, b)
                    break


def CheckRoiNum(data_folder):
    from DataProcess.MaxRoi import KeepLargest
    case_list = os.listdir(data_folder)
    for case in case_list:
        volume_list = []
        # path
        case_path = os.path.join(data_folder, case)
        t2_path = os.path.join(case_path, 't2.nii')
        roi_path = os.path.join(case_path, 'roi.nii')

        # Load data
        _, t2, _ = LoadNiiData(t2_path, dtype=np.float32)

        _, roi, _ = LoadNiiData(roi_path, dtype=np.uint8)

        label_im, nb_labels = ndimage.label(roi)
        if nb_labels != 1:
            for index in range(1, nb_labels + 1):
                volume = (label_im == index).sum()
                if volume != 1:
                    volume_list.append(volume)
            if len(volume_list) != 1:
                print(case, volume_list)


def Show():
    plt.suptitle(case)
    plt.subplot(221)
    plt.title('t2')
    plt.imshow(t2_slice, cmap='gray')
    plt.contour(roi_slice, colors='r')
    plt.axis('off')
    plt.subplot(222)
    plt.title('dwi')
    plt.imshow(dwi_slice, cmap='gray')
    plt.contour(roi_slice, colors='r')
    plt.axis('off')
    plt.subplot(223)
    plt.title('adc')
    plt.imshow(adc_slice, cmap='gray')
    plt.contour(roi_slice, colors='r')
    plt.axis('off')
    plt.savefig(os.path.join(store_path, case + '.jpg'))
    plt.close()
    plt.show()


if __name__ == '__main__':
    data_folder = resample_folder
    save_path =cnn_folder
    WriteH5(data_folder, save_path)
    # TestWhiteH5()


