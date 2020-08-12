import os
import numpy as np
import pandas as pd

from BasicTool.MeDIT.SaveAndLoad import LoadImage

data_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
store_path = r'C:\Users\ZhangYihong\Desktop\ProstateVolume.csv'
case_list = os.listdir(data_folder)
for case in case_list:
    case_path = os.path.join(data_folder, case)
    prostate_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
    image, data, ref = LoadImage(prostate_path)
    resolution = image.GetSpacing()
    pixels = np.sum(data)
    volume = pixels*resolution[0]*resolution[1]*resolution[2]
    print(case)

    # one_label_df = pd.DataFrame(volume, index=[case], columns=['volume'])
    # if case == case_list[0]:
    #     one_label_df.to_csv(store_path, mode='a+', header=True)
    # else:
    #     one_label_df.to_csv(store_path, mode='a+', header=False)