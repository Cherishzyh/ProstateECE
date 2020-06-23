import os
import matplotlib.pyplot as plt
from BasicTool.MeDIT.Visualization import Imshow3DArray
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.SaveAndLoad import LoadNiiData
import numpy as np


def NPY():
    t2_case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\T2Slice\Train'
    prostate_case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\ProstateSlice\Train'
    roi_case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\RoiSlice\Train'
    case_list = os.listdir(t2_case_folder)
    for case in case_list:

        t2_path = os.path.join(t2_case_folder, case)
        prostate_path = os.path.join(prostate_case_folder, case)
        roi_path = os.path.join(roi_case_folder, case)

        t2_data = np.transpose(np.load(t2_path), [1, 2, 0])
        roi_data = np.transpose(np.load(roi_path), [1, 2, 0])
        prostate_data = np.transpose(np.load(prostate_path), [1, 2, 0])

        if len(list(np.unique(prostate_data))) == 1:
            print(case)
        # print(case, np.unique(roi_data))
        # plt.imshow(np.squeeze(t2_data), cmap='gray')
        # plt.contour(np.squeeze(prostate_data), colors='r')
        # plt.contour(np.squeeze(roi_data), colors='y')
        # plt.show()
NPY()

def NII():
    case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case_list = os.listdir(case_folder)
    case = 'FAN DA HAI'
    case_path = os.path.join(case_folder, case)
    t2_path = os.path.join(case_path, 't2.nii')
    prostate_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
    cancer_path = os.path.join(case_path, 'roi.nii')

    _, t2_data, _ = LoadNiiData(t2_path)
    _, prostate_data, _ = LoadNiiData(prostate_path)
    _, cancer_data, _ = LoadNiiData(cancer_path)

    Imshow3DArray(Normalize01(t2_data), roi=[Normalize01(prostate_data), Normalize01(cancer_data)])
# NII()