import os
import numpy as np
import shutil
import SimpleITK as sitk

from MeDIT.SaveAndLoad import LoadImage, SaveNiiImage
from MeDIT.ImageProcess import GetImageFromArrayByImage
from MeDIT.Others import CopyFile
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01


def Normalize(process_folder, resampe_folder):
    for case in os.listdir(process_folder):
        case_folder = os.path.join(process_folder, case)
        store_case_folder = os.path.join(resampe_folder, case)
        if not os.path.exists(store_case_folder):
            os.mkdir(store_case_folder)

        print('Normalizing {}'.format(case))
        roi_path = os.path.join(case_folder, 'roi.nii')
        if not os.path.exists(roi_path):
            continue

        roi_image, _, roi_data = LoadImage(roi_path, dtype=np.uint8)

        if list(np.unique(roi_data)) == [0, 255]:
            roi = np.array(roi_data / 255, dtype=np.uint8)
        else:
            roi = roi_data
        roi_array = GetImageFromArrayByImage(roi, roi_image)
        SaveNiiImage(os.path.join(store_case_folder, 'normalize_roi.nii'), roi_array)
        if os.path.exists(os.path.join(case_folder, 'normalize_roi.nii')):
            os.remove(os.path.join(case_folder, 'normalize_roi.nii'))


# Normalize()


def TestNormalize():
    # cg
    cg_path = r'X:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii\2019-CA-formal-QIAN SHENG KUN\ROI_jkw0.nii'
    # wg
    wg_path = r'X:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii\2019-CA-formal-QIAN SHENG KUN\ROI_jkw1.nii'
    t2_path = r'X:\RawData\TZ_ROI_20191119\prostate segmentation _PZ_TZ\Nii\2019-CA-formal-QIAN SHENG KUN\t2_Resize.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadImage(t2_path)
    _, _, cg = LoadImage(cg_path, dtype=np.uint8)
    _, _, wg = LoadImage(wg_path, dtype=np.uint8)


    Imshow3DArray(Normalize01(t2), roi=[Normalize01(cg), Normalize01(wg)])

# TestNormalize()


########################################################


def ResampleData():
    from MIP4AIM.NiiProcess.Resampler import Resampler

    root_folder = r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath'
    dest_root = r'X:\PrcoessedData\BCR-ECE-score\ResampelNow'

    resampler = Resampler()
    # for case in os.listdir(root_folder):
    for case in ['CSJ^chen shi jie', 'WU XIAO LEI', 'WXZ^wu xi zhong', 'XSJ^xu shou jun']:
        if case == "NoPath":
            continue
        case_folder = os.path.join(root_folder, case)

        if not os.path.isdir(case_folder):
            continue

        dest_case_folder = os.path.join(dest_root, case)
        if not os.path.exists(dest_case_folder):
            os.mkdir(dest_case_folder)

        print('Resample of {}'.format(case))
        # t2_path = os.path.join(case_folder, 't2.nii')
        # roi_path = os.path.join(case_folder, 'ROI_CK0_1.nii.gz')
        # prostate_path = os.path.join(case_folder, 'ProstateROI_TrumpetNet.nii.gz')
        # dwi_path = os.path.join(case_folder, 'dwi_Reg.nii')
        # adc_path = os.path.join(case_folder, 'adc_Reg.nii')
        t2_path = os.path.join(case_folder, 't2.nii')
        roi_path = os.path.join(case_folder, 'roi.nii')
        prostate_path = os.path.join(case_folder, 'ProstateROI_TrumpetNet.nii.gz')
        dwi_path = os.path.join(case_folder, 'dwi_b1500_Reg.nii')
        adc_path = os.path.join(case_folder, 'adc_Reg.nii')

        if not os.path.exists(t2_path) or not os.path.exists(roi_path) or not os.path.exists(dwi_path) \
                or not os.path.exists(adc_path) or not os.path.exists(prostate_path):
            print('not path')
            continue
        try:
            t2_image, t2_data, _ = LoadImage(t2_path, is_show_info=False)
            dwi_image, dwi_data, _ = LoadImage(dwi_path, is_show_info=False)
            adc_image, adc_data, _ = LoadImage(adc_path, is_show_info=False)
            roi_image, roi_data, _ = LoadImage(roi_path, dtype=np.uint8, is_show_info=False)
            pro_image, pro_data, _ = LoadImage(prostate_path, dtype=np.uint8, is_show_info=False)

            resampler.ResizeSipmleITKImage(t2_image, expected_resolution=[0.5, 0.5, -1],
                                           store_path=os.path.join(dest_case_folder, 't2_5x5.nii.gz'))
            resampler.ResizeSipmleITKImage(dwi_image, expected_resolution=[0.5, 0.5, -1],
                                           store_path=os.path.join(dest_case_folder, 'dwi_Reg_5x5.nii.gz'))
            resampler.ResizeSipmleITKImage(adc_image, expected_resolution=[0.5, 0.5, -1],
                                           store_path=os.path.join(dest_case_folder, 'adc_Reg_5x5.nii.gz'))
            resampler.ResizeSipmleITKImage(roi_image, is_roi=True, expected_resolution=[0.5, 0.5, -1],
                                           store_path=os.path.join(dest_case_folder, 'roi_5x5.nii.gz'))
            resampler.ResizeSipmleITKImage(pro_image, is_roi=True, expected_resolution=[0.5, 0.5, -1],
                                           store_path=os.path.join(dest_case_folder, 'pro_5x5.nii.gz'))
        except Exception as e:
            print(case, e)
        # shutil.copy(os.path.join(case_folder, 'roi.csv'), os.path.join(dest_case_folder, 'roi.csv'))

ResampleData()

def TestResampleData():
    roi_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\roi.nii'
    t2_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\t2.nii'
    dwi_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\dwi.nii'
    adc_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\adc.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadImage(t2_path, is_show_info=True)
    _, _, dwi = LoadImage(dwi_path, is_show_info=True)
    img, _, adc = LoadImage(adc_path, is_show_info=True)
    _, _, roi = LoadImage(roi_path, dtype=np.uint8, is_show_info=True)
    #
    new_dwi = dwi[..., 2]
    new_img = sitk.GetImageFromArray(new_dwi)
    new_img.SetDirection(img.GetDirection())
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetOrigin(img.GetOrigin())
    sitk.WriteImage(new_img, os.path.join(r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian', 'max_dwi.nii'))
    #
    Imshow3DArray(Normalize01(t2), roi=Normalize01(roi))
    Imshow3DArray(Normalize01(dwi))
    # Imshow3DArray(Normalize01(adc))


# TestResampleData()


######################################################
# for case in os.listdir(r'X:\PrcoessedData\BCR-ECE-score'):
#     case_folder = os.path.join(r'X:\PrcoessedData\BCR-ECE-score', case)
#     if len(os.listdir(case_folder)) < 1:
#         shutil.copytree(os.path.join(r'X:\RawData\BCR-ECE-score\BCR-ECE-score', case), os.path.join(r'C:\Users\ZhangYihong\Desktop\aaaa', case))


# for case in os.listdir(r'C:\Users\ZhangYihong\Desktop\aaaa\NoPath'):
#     if case not in os.listdir(r'X:\StoreFormatData\ProstateCancerECE\ResampleData'):
#         print(case)
#
# t2_1 = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\RawData\BCR-ECE-score\BCR-ECE-score\CSJ^chen shi jie\dwi.nii'))
# t2_2 = sitk.GetArrayFromImage(sitk.ReadImage(r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath\CSJ^chen shi jie\dwi.nii'))
# print((t2_2 == t2_1).all())