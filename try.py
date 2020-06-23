import numpy as np
import os

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
feature_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/t2.npy'
t2_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/T2Slice/Test/BHX^bao han xiu ^^6875-5_slice9.npy'
data = np.load(feature_path)
data = np.squeeze(data)
t2_data = np.load(t2_path)
t2_data = np.squeeze(t2_data)
plt.subplot(121)
plt.imshow(data, cmap='gray')
plt.subplot(122)
plt.imshow(t2_data, cmap='gray')
plt.show()
