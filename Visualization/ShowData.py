import os
from BasicTool.MeDIT.Visualization import Imshow3DArray
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.SaveAndLoad import LoadNiiData

case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
case_list = os.listdir(case_folder)
for case in case_list:
    case_path = os.path.join(case_folder, case)
    t2_path = os.path.join(case_path, 't2.nii')
    prostate_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
    cancer_path = os.path.join(case_path, 'roi.nii')

    _, _, t2_data = LoadNiiData(t2_path)
    _, _, prostate_data = LoadNiiData(prostate_path)
    _, _, cancer_data = LoadNiiData(cancer_path)

    Imshow3DArray(Normalize01(t2_data), roi=[Normalize01(prostate_data), Normalize01(cancer_data)])
