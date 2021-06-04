import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import shutil

from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Normalize import Normalize01, NormalizeZ
from BasicTool.MeDIT.Visualization import Imshow3DArray
from BasicTool.MeDIT.ArrayProcess import ExtractPatch

from ECEDataProcess.DataProcess.MaxRoi import SelectMaxRoiSlice, GetRoiCenter, KeepLargest

# info = pd.read_csv(csv_path, usecols=['case', 'pECE'], index_col=['case'])


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
    case_list = os.listdir(data_folder)
    crop_shape = (1, 280, 280)

    for case in case_list:
        # path
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

        dataname = case + '_slice' + str(slice) + '.h5'
        datapath = os.path.join(save_path, dataname)

        with h5py.File(datapath, 'w') as f:
            f['input_0'] = t2_slice_3d
            f['input_1'] = dwi_slice_3d
            f['input_2'] = adc_slice_3d
            f['output_0'] = roi_slice_3d
            f['output_1'] = ece


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
    from ECEDataProcess.DataProcess.SelectDWI import NearTrueB
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
    from ECEDataProcess.DataProcess.MaxRoi import KeepLargest
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


def CorrectName():
    data_root = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\Test'
    folder_list = ['AdcSlice', 'DwiSlice', 'T2Slice', 'ProstateSlice', 'PCaSlice', 'DistanceMap']
    folder_list_new = ['AdcSliceNew', 'DwiSliceNew', 'T2SliceNew', 'ProstateSliceNew', 'PCaSliceNew', 'DistanceMapNew']

    for index, folder in enumerate(folder_list):
        data_folder = os.path.join(data_root, folder)
        new_data_folder = os.path.join(data_root, folder_list_new[index])
        if not os.path.exists(new_data_folder):
            os.mkdir(new_data_folder)
        print('###########copy {}#############'.format(folder))
        for j, case in enumerate(os.listdir(data_folder)):
            name, slice = case.split('_')
            new_name = name + '_-_' + slice
            print(new_name)
            shutil.copy(os.path.join(data_folder, case), os.path.join(new_data_folder, new_name))


def SaveNPY(data_folder, save_path):
    case_list = os.listdir(data_folder)
    # crop_shape = (1, 280, 280)

    for case in case_list:
        case = '601548'
        # path
        case_path = os.path.join(data_folder, case)
        ct_path = os.path.join(case_path, 'data.nii.gz')
        pca_path = os.path.join(case_path, 'roi.nii.gz')
        pro_path = os.path.join(case_path, 'only_kidney_roi_lq.nii.gz')

        try:
            _, ct, _ = LoadImage(ct_path, dtype=np.float32)
            _, pro, _ = LoadImage(pro_path, dtype=np.uint8)
            _, pca, _ = LoadImage(pca_path, dtype=np.uint8)

            slice = SelectMaxRoiSlice(pca)
            # slice = 8

            ct_slice_3d = ct[np.newaxis, ..., slice]
            pro_slice_3d = pro[np.newaxis, ..., slice]
            pca_slice_3d = pca[np.newaxis, ..., slice]

            pro_slice_3d = np.clip(pro_slice_3d, a_min=0, a_max=1)

            plt.imshow(np.squeeze(ct_slice_3d), cmap='gray')
            plt.contour(np.squeeze(pro_slice_3d), colors='r')
            plt.contour(np.squeeze(pca_slice_3d), colors='y')
            plt.show()
            plt.close()

            dataname = case + '_-_slice' + str(slice) + '.npy'
            ct_save_path = os.path.join(save_path, 'ct_slice/{}'.format(dataname))
            cancer_save_path = os.path.join(save_path, 'cancer_slice/{}'.format(dataname))
            kindey_save_path = os.path.join(save_path, 'kindey_slice/{}'.format(dataname))

            np.save(ct_save_path, ct_slice_3d)
            np.save(cancer_save_path, pca_slice_3d)
            np.save(kindey_save_path, pro_slice_3d)
        except Exception as e:
            print(e)
        break


def CheckROINum(data_folder):
    roi_folder = os.path.join(data_folder, 'cancer_slice')
    ct_folder = os.path.join(data_folder, 'ct_slice')
    gland_folder = os.path.join(data_folder, 'kindey_slice')
    for case in os.listdir(roi_folder):
        roi = np.squeeze(np.load(os.path.join(roi_folder, case)))
        # gland = np.squeeze(np.load(os.path.join(gland_folder, case)))
        # ct = np.squeeze(np.load(os.path.join(ct_folder, case)))
        _, num, new_roi = KeepLargest(roi)
        if num > 1:
            print(case)
            # np.save(os.path.join(roi_folder, case), new_roi)


def CropKeepLargest():
    data_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy'
    roi_folder = os.path.join(data_folder, 'cancer_slice')
    gland_folder = os.path.join(data_folder, 'kindey_slice')
    ct_folder = os.path.join(data_folder, 'ct_slice')
    shape = (300, 300)
    for case in os.listdir(roi_folder):
        case = '601548_-_slice8.npy'
        ct = np.squeeze(np.load(os.path.join(ct_folder, case)))
        roi = np.squeeze(np.load(os.path.join(roi_folder, case)))
        gland = np.squeeze(np.load(os.path.join(gland_folder, case)))

        center, _ = GetRoiCenter(roi)

        roi_new, _ = ExtractPatch(roi, patch_size=shape, center_point=center)
        gland_new, _ = ExtractPatch(gland, patch_size=shape, center_point=center)
        ct_new, _ = ExtractPatch(ct, patch_size=shape, center_point=center)

        _, _, gland_new = KeepLargest(gland_new)

        ct_slice_3d = ct_new[np.newaxis, ...]
        gland_slice_3d = gland_new[np.newaxis, ...]
        roi_slice_3d = roi_new[np.newaxis, ...]

        plt.subplot(121)
        plt.imshow(ct, cmap='gray')
        plt.contour(gland, colors='r')
        plt.contour(roi, colors='y')
        plt.scatter(x=center[1], y=center[0])
        plt.subplot(122)
        plt.imshow(ct_new, cmap='gray')
        plt.contour(gland_new, colors='r')
        plt.contour(roi_new, colors='y')
        plt.scatter(x=150, y=150)
        plt.show()

        np.save(os.path.join(ct_folder, case), ct_slice_3d)
        np.save(os.path.join(roi_folder, case), roi_slice_3d)
        np.save(os.path.join(gland_folder, case), gland_slice_3d)
        break


def Normailzation():
    data_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy'
    roi_folder = os.path.join(data_folder, 'cancer_slice')
    gland_folder = os.path.join(data_folder, 'kindey_slice')
    ct_folder = os.path.join(data_folder, 'ct_slice')

    for case in os.listdir(roi_folder):
        ct = np.load(os.path.join(ct_folder, case))
        roi = np.load(os.path.join(roi_folder, case))
        gland = np.load(os.path.join(gland_folder, case))

        ct = NormalizeZ(ct)
        if (np.unique(roi) == np.array([0, 1])).all() and (np.unique(gland) == np.array([0, 1])).all():
            continue
        else:
            print(case)
        np.save(os.path.join(ct_folder, case), ct)


def Add():
    data_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy'
    atten_folder = os.path.join(data_folder, 'atten_slice')

    for case in os.listdir(atten_folder):
        atten = np.load(os.path.join(atten_folder, case))

        atten = atten[np.newaxis, ...]

        np.save(os.path.join(atten_folder, case), atten)


if __name__ == '__main__':
    data_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey901_new'
    save_path = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy'

    # SaveNPY(data_folder, save_path)
    # CheckROINum(save_path)
    # CropKeepLargest()
    # Normailzation()
    Add()

