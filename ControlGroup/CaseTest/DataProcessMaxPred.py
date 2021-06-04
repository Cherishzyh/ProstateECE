import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

from SSHProject.BasicTool.MeDIT.SaveAndLoad import LoadImage
from SSHProject.BasicTool.MeDIT.Normalize import Normalize01
from SSHProject.BasicTool.MeDIT.Visualization import Imshow3DArray

from ECEDataProcess.DataProcess.H5 import CropT2Data, CropRoiData, Checkb, CheckRoiNum
from ECEDataProcess.DataProcess.MaxRoi import SelectMaxRoiSlice, GetRoiCenter, KeepLargest
from ControlGroup.CaseTest.Modify import *
from DistanceMap.RoiDistanceMap import FindRegion

# MakeH5()
def WriteNPY(data_folder, data_type='train'):
    crop_shape = (1, 280, 280)
    if data_type == 'train':
        pred_path = r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/PAGNet_RightCrop/Train'
        save_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Train'
    else:
        pred_path = r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/PAGNet_RightCrop/Test'
        save_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Test'

    case_list = os.listdir(pred_path)
    case_list = [case[: case.index('.h5')] for case in case_list]
    for case in sorted(case_list):
        # path
        case_path = os.path.join(data_folder, case)
        t2_path = os.path.join(case_path, 't2.nii')
        dwi_path = os.path.join(case_path, 'dwi_Reg.nii')
        adc_path = os.path.join(case_path, 'adc_Reg.nii')
        roi_path = os.path.join(case_path, 'roi.nii')
        prostate_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')

        # save_path
        t2_save_path = os.path.join(save_folder, 'T2Slice')
        dwi_save_path = os.path.join(save_folder, 'DwiSlice')
        adc_save_path = os.path.join(save_folder, 'AdcSlice')
        pca_save_path = os.path.join(save_folder, 'PCaSlice')
        prostate_save_path = os.path.join(save_folder, 'ProstateSlice')
        distance_save_path = os.path.join(save_folder, 'DistanceMap')

        # Load data
        _, t2, _ = LoadImage(t2_path, dtype=np.float32)
        _, dwi, _ = LoadImage(dwi_path, dtype=np.float32)
        _, adc, _ = LoadImage(adc_path, dtype=np.float32)
        _, pca, _ = LoadImage(roi_path, dtype=np.uint8)
        _, prostate, _ = LoadImage(prostate_path, dtype=np.uint8)

        t2 = t2.transpose((2, 0, 1))
        dwi = dwi.transpose((2, 0, 1))
        adc = adc.transpose((2, 0, 1))
        pca = pca.transpose((2, 0, 1))
        prostate = prostate.transpose((2, 0, 1))

        _, _, new_roi = KeepLargest(pca)

        slice_list_pca = GetROISlice(pca)
        slice_list_pro = GetROISlice(prostate)
        slice_list = [slice for slice in slice_list_pca if slice in slice_list_pro]
        slice_preds, label = Get3DPred(case, pred_path, if_negative=True)
        slice = slice_list[slice_preds.index(max(slice_preds))]

        center = GetRoiCenterBefore(new_roi[slice, ...])

        t2_slice = CropT2Data(t2, crop_shape, slice, center=center)
        dwi_slice = CropT2Data(dwi, crop_shape, slice, center=center)
        adc_slice = CropT2Data(adc, crop_shape, slice, center=center)
        roi_slice = CropRoiData(new_roi, crop_shape, slice, center=center)
        prostate_slice = CropRoiData(prostate, crop_shape, slice, center=center)
        distance_map = FindRegion(prostate_slice, roi_slice)
        #
        # plt.subplot(121)
        # plt.imshow(t2_slice, cmap='gray')
        # plt.contour(roi_slice, colors='r')
        # plt.contour(prostate_slice, colors='y')
        # plt.subplot(122)
        # plt.imshow(distance_map, cmap='jet')
        # plt.show()

        t2_slice_3d = t2_slice[np.newaxis, ...]
        dwi_slice_3d = dwi_slice[np.newaxis, ...]
        adc_slice_3d = adc_slice[np.newaxis, ...]
        roi_slice_3d = roi_slice[np.newaxis, ...]
        prostate_slice_3d = prostate_slice[np.newaxis, ...]
        distance_map_3d = distance_map[np.newaxis, ...]

        np.save(os.path.join(t2_save_path, case+'_-_slice'+str(slice)+'.npy'), t2_slice_3d)
        np.save(os.path.join(dwi_save_path, case+'_-_slice'+str(slice)+'.npy'), dwi_slice_3d)
        np.save(os.path.join(adc_save_path, case+'_-_slice'+str(slice)+'.npy'), adc_slice_3d)
        np.save(os.path.join(pca_save_path, case+'_-_slice'+str(slice)+'.npy'), roi_slice_3d)
        np.save(os.path.join(prostate_save_path, case+'_-_slice'+str(slice)+'.npy'), prostate_slice_3d)
        np.save(os.path.join(distance_save_path, case + '_-_slice' + str(slice) + '.npy'), distance_map_3d)
        print(case)


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
        _, _, t2 = LoadImage(t2_path, dtype=np.float32)
        _, _, dwi = LoadImage(dwi_path, dtype=np.float32)
        _, _, adc = LoadImage(adc_path, dtype=np.float32)
        _, _, roi = LoadImage(roi_path, dtype=np.uint8)

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


if __name__ == '__main__':
    data_folder = r'/home/zhangyihong/Documents/ProstateECE/ResampleData'
    # save_path = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred'
    WriteNPY(data_folder, data_type='train')
    # TestWhiteH5()
    #
    # info = pd.read_csv(csv_path, usecols=['case', 'pECE'], index_col=['case'])


