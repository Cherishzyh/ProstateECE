import os
import numpy as np
import matplotlib.pyplot as plt

data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY/DistanceMap/Validation'
des_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY/DistanceMap0.2/Validation'

case_list = os.listdir(data_folder)
for case in case_list:
    dis_map = np.load(os.path.join(data_folder, case))
    dis_map_norm = dis_map * 0.8 + 0.2
    # np.save(os.path.join(des_folder, case), dis_map_norm)
    plt.subplot(121)
    plt.imshow(np.squeeze(dis_map), cmap='jet', vmin=0., vmax=1.)
    plt.subplot(122)
    plt.imshow(np.squeeze(dis_map_norm), cmap='jet', vmin=0., vmax=1.)
    plt.show()



