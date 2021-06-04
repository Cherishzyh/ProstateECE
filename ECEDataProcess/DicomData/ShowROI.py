import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from BasicTool.MeDIT.Visualization import Imshow3DArray
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.SaveAndLoad import LoadImage


def Show():
    data_folder = r'X:\RawData\Kindey901_new'
    # for case in os.listdir(data_folder):
    case = '1286125'
    case_folder = os.path.join(data_folder, case)
    image_path = os.path.join(case_folder, 'data.nii.gz')
    pro_path = os.path.join(case_folder, 'only_kidney_roi_lq.nii.gz')
    pca_path = os.path.join(case_folder, 'roi.nii.gz')

    _, image, _ = LoadImage(image_path)
    _, pro, _ = LoadImage(pro_path)
    _, pca, _ = LoadImage(pca_path)

    Imshow3DArray(Normalize01(image), roi=[Normalize01(pro), Normalize01(pca)])

Show()