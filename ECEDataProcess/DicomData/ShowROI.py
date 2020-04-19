import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01
from MeDIT.SaveAndLoad import LoadNiiData

path = r'X:\PrcoessedData\ProstateCancerECE\BSL^bai song lai ^^6698-8'

t2_path = os.path.join(path, 't2.nii')
roi_path = os.path.join(path, 'roi.nii')
# roi0_path = os.path.join(path, 'roi0.nii')
# roi1_path = os.path.join(path, 'roi1.nii')
# roi2_path = os.path.join(path, 'roi2.nii')
dwi_path = os.path.join(path, 'dki.nii')
adc_path = os.path.join(path, 'adc.nii')
_, t2_array, _ = LoadNiiData(t2_path)
_, roi_array, _ = LoadNiiData(roi_path)
# _, roi0_array, _ = LoadNiiData(roi0_path)
# _, roi1_array, _ = LoadNiiData(roi1_path)
# _, roi2_array, _ = LoadNiiData(roi2_path)
_, dwi_array, _ = LoadNiiData(dwi_path)
_, adc_array, _ = LoadNiiData(adc_path)

# t2 = sitk.ReadImage(t2_path)
# t2_array = sitk.GetArrayFromImage(t2, )
#
# roi = sitk.ReadImage(roi_path)
# roi_array = sitk.GetArrayFromImage(roi)
#
t2_array = np.transpose(t2_array, (1, 2, 0))
roi_array = np.transpose(roi_array, (1, 2, 0))
# roi0_array = np.transpose(roi0_array, (1, 2, 0))
# roi1_array = np.transpose(roi1_array, (1, 2, 0))
# roi2_array = np.transpose(roi2_array, (1, 2, 0))
dwi_array = dwi_array[1, ...]
dwi_array = np.transpose(dwi_array, (1, 2, 0))
# adc_array = np.transpose(adc_array, (1, 2, 0))
Imshow3DArray(Normalize01(t2_array), roi=Normalize01(roi_array))
Imshow3DArray(Normalize01(dwi_array))
# Imshow3DArray(Normalize01(t2_array), roi=[Normalize01(roi0_array), Normalize01(roi1_array), Normalize01(roi2_array)])
# Imshow3DArray(Normalize01(adc_array))


