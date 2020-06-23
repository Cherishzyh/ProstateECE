import os
import numpy as np

from MIP4AIM.NiiProcess.Registrator import Registrator
from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Normalize import Normalize01
from MeDIT.Visualization import Imshow3DArray

from FilePath import process_folder

registrator = Registrator()

def RegistrateBySpacing(case_folder):
    t2_path = os.path.join(case_folder, 't2.nii')
    adc_path = os.path.join(case_folder, 'adc.nii')
    dwi_path = os.path.join(case_folder, 'max_b_dwi.nii')

    registrator.fixed_image = t2_path

    registrator.moving_image = adc_path
    try:
        registrator.RegistrateBySpacing(store_path=registrator.GenerateStorePath(adc_path))
    except:
        return False, 'Align ADC Failed'

    registrator.moving_image = dwi_path
    try:
        registrator.RegistrateBySpacing(store_path=registrator.GenerateStorePath(dwi_path))
    except:
        return False, 'Align DWI Failed'

    return True, ''


def Path(case_folder):
    t2_path = os.path.join(case_folder, 't2.nii')
    roi_path = os.path.join(case_folder, 'roi.nii')
    adc_path = os.path.join(case_folder, 'adc_Reg.nii')
    dwi_path = os.path.join(case_folder, 'max_b_dwi_Reg.nii')

    _, _, t2 = LoadNiiData(t2_path, is_show_info=True)
    _, _, dwi = LoadNiiData(dwi_path, is_show_info=True)
    _, _, adc = LoadNiiData(adc_path, is_show_info=True)
    _, _, roi = LoadNiiData(roi_path, dtype=np.uint8, is_show_info=True)

    Imshow3DArray(Normalize01(t2), roi=Normalize01(roi))
    Imshow3DArray(Normalize01(dwi))
    Imshow3DArray(Normalize01(adc))


if __name__ == '__main__':
    # case_folder = r'C:\Users\ZhangYihong\Desktop\try\BAO ZHENG LI'
    case_folder = r'X:\PrcoessedData\ProstateCancerECE\CSJ^chen shi jie'
    RegistrateBySpacing(case_folder)
    # case_list = os.listdir(process_folder)
    # for case in case_list:
    #     case_folder = os.path.join(process_folder, case)
    #     try:
    #         RegistrateBySpacing(case_folder)
    #     except Exception as e:
    #         print(case, e)
    # Path(case_folder)
