import numpy as np
import os
import shutil

import matplotlib.pyplot as plt


# feature_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map'
# feature_list = os.listdir(feature_path)
# feature_list.sort()
# data = []
# for case in feature_list:
#     npy_case = os.path.join(feature_path, case)
#     data = (np.load(npy_case))
#     data = np.squeeze(data)
#     plt.imshow(data, cmap='gray')
#     plt.show()
# feature_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/t2.npy'
# t2_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/T2Slice/Test/BHX^bao han xiu ^^6875-5_slice9.npy'
# data = np.load(feature_path)
# data = np.squeeze(data)
# t2_data = np.load(t2_path)
# t2_data = np.squeeze(t2_data)
# plt.subplot(121)
# plt.imshow(data, cmap='gray')
# plt.subplot(122)
# plt.imshow(t2_data, cmap='gray')
# plt.show()

case_name = 'DSR^dai shou rong_slice16.npy'
case_name_before = 'DSR^dai shou rong_slice14.npy'

data_path = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain'
data_path_list = os.listdir(data_path)
for folder in data_path_list:
    case_folder = os.path.join(os.path.join(data_path, folder), 'PreTrain')
    case_path_before = os.path.join(case_folder, case_name_before)
    case_path = os.path.join(case_folder, case_name)
    if os.path.exists(case_path_before):

        shutil.copy(case_path_before, case_path)
        print(case_path_before)
        print(case_path)
        os.remove(case_path_before)


