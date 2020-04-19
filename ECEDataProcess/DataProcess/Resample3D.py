import os
import numpy as np
import shutil
import SimpleITK as sitk

from MeDIT.SaveAndLoad import LoadNiiData, SaveNiiImage
from MeDIT.ImageProcess import GetImageFromArrayByImage
from MeDIT.Others import CopyFile
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01

from FilePath import resample_folder, process_folder, desktop_path


def Normalize():
    for case in os.listdir(process_folder):
        case_folder = os.path.join(process_folder, case)
        store_case_folder = os.path.join(resampe_folder, case)
        if not os.path.exists(store_case_folder):
            os.mkdir(store_case_folder)

        print('Normalizing {}'.format(case))
        roi_path = os.path.join(case_folder, 'roi.nii')
        if not os.path.exists(roi_path):
            continue

        roi_image, _, roi_data = LoadNiiData(roi_path, dtype=np.uint8)

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

    _, _, t2 = LoadNiiData(t2_path)
    _, _, cg = LoadNiiData(cg_path, dtype=np.uint8)
    _, _, wg = LoadNiiData(wg_path, dtype=np.uint8)


    Imshow3DArray(Normalize01(t2), roi=[Normalize01(cg), Normalize01(wg)])

# TestNormalize()


########################################################


def ResampleData():
    from MIP4AIM.NiiProcess.Resampler import Resampler

    root_folder = process_folder
    dest_root = r'X:\StoreFormatData\ProstateCancerECE\FailedData'

    resampler = Resampler()
    for case in os.listdir(root_folder):
        case = 'CSJ^chen shi jie'
        case_folder = os.path.join(root_folder, case)

        if not os.path.isdir(case_folder):
            continue

        dest_case_folder = os.path.join(dest_root, case)
        if not os.path.exists(dest_case_folder):
            os.mkdir(dest_case_folder)

        print('Resample of {}'.format(case))
        t2_path = os.path.join(case_folder, 't2.nii')
        roi_path = os.path.join(case_folder, 'roi.nii')
        dwi_path = os.path.join(case_folder, 'max_b_dwi_Reg.nii')
        adc_path = os.path.join(case_folder, 'adc_Reg.nii')
        if not os.path.exists(t2_path) or not os.path.exists(roi_path) or not os.path.exists(dwi_path) or not os.path.exists(adc_path):
            print('not path')
            continue
        t2_image, _, t2_data = LoadNiiData(t2_path, is_show_info=False)
        dwi_image, _, dwi_data = LoadNiiData(dwi_path, is_show_info=False)
        adc_image, _, adc_data = LoadNiiData(adc_path, is_show_info=False)
        roi_image, _, roi_data = LoadNiiData(roi_path, dtype=np.uint8, is_show_info=False)

        resampler.ResizeSipmleITKImage(t2_image, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 't2.nii'))
        resampler.ResizeSipmleITKImage(dwi_image, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 'dwi_Reg.nii'))
        resampler.ResizeSipmleITKImage(adc_image, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 'adc_Reg.nii'))
        resampler.ResizeSipmleITKImage(roi_image, is_roi=True, expected_resolution=[0.5, 0.5, -1],
                                       store_path=os.path.join(dest_case_folder, 'roi.nii'))
        shutil.copy(os.path.join(case_folder, 'roi.csv'), os.path.join(dest_case_folder, 'roi.csv'),)
        break

# ResampleData()

def TestResampleData():
    roi_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\roi.nii'
    t2_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\t2.nii'
    dwi_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\dwi.nii'
    adc_path = r'X:\PrcoessedData\ProstateCancerECE\ZJ^zhang jian\adc.nii'

    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01

    _, _, t2 = LoadNiiData(t2_path, is_show_info=True)
    _, _, dwi = LoadNiiData(dwi_path, is_show_info=True)
    img, _, adc = LoadNiiData(adc_path, is_show_info=True)
    _, _, roi = LoadNiiData(roi_path, dtype=np.uint8, is_show_info=True)
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


TestResampleData()


########################################################

