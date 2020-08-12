import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

dis_map_path = r'/home/zhangyihong/Documents/ProstateECE/NPY/DistanceMap0.2/Test'
dis_map_list = os.listdir(dis_map_path)
for map in dis_map_list:
    dis_map_data = np.load(os.path.join(dis_map_path, map))
    dis_map_data = torch.tensor(dis_map_data[np.newaxis, ...], dtype=torch.float)
    #
    #
    # avg_out = torch.mean(dis_map_data, dim=1, keepdim=True)
    # max_out, _ = torch.max(dis_map_data, dim=1, keepdim=True)
    #
    # out = torch.cat([avg_out, max_out], dim=1)
    #
    # conv_map = nn.Conv2d(2, 1, 7, padding=3, bias=False)(out)
    # # nn.Conv2d(2, 1, 7, padding=3, bias=False)
    # conv_map_sig = nn.Sigmoid()(conv_map)
    #
    # plt.subplot(231)
    # plt.imshow(np.squeeze(dis_map_data), cmap='jet', vmin=0., vmax=1.)
    # plt.subplot(232)
    # plt.imshow(np.squeeze(conv_map.sigmoid().detach().numpy()), cmap='jet', vmin=0., vmax=1.)
    # plt.subplot(233)
    # plt.imshow(np.squeeze(conv_map_sig.sigmoid().detach().numpy()), cmap='jet', vmin=0., vmax=1.)
    # plt.subplot(234)
    # plt.imshow(np.squeeze(max_out), cmap='jet', vmin=0., vmax=1.)
    # plt.subplot(235)
    # plt.imshow(np.squeeze(avg_out), cmap='jet', vmin=0., vmax=1.)
    # # plt.subplot(236)
    # # plt.imshow(np.squeeze(out), cmap='jet', vmin=0., vmax=1.)
    # plt.colorbar()
    # plt.show()
    dis_map_92 = F.interpolate(dis_map_data, size=(92, 92), mode='nearest')
    dis_map_46 = F.interpolate(dis_map_data, size=(46, 46), mode='nearest')
    dis_map_23 = F.interpolate(dis_map_data, size=(23, 23), mode='nearest')
    dis_map_12 = F.interpolate(dis_map_data, size=(12, 12), mode='nearest')

    plt.subplot(231)
    plt.imshow(np.squeeze(dis_map_data), cmap='jet', vmin=0., vmax=1.)
    plt.subplot(232)
    plt.imshow(np.squeeze(dis_map_92), cmap='jet', vmin=0., vmax=1.)
    plt.subplot(233)
    plt.imshow(np.squeeze(dis_map_46), cmap='jet', vmin=0., vmax=1.)
    plt.subplot(234)
    plt.imshow(np.squeeze(dis_map_23), cmap='jet', vmin=0., vmax=1.)
    plt.subplot(235)
    plt.imshow(np.squeeze(dis_map_12), cmap='jet', vmin=0., vmax=1.)
    # plt.subplot(236)
    # plt.imshow(np.squeeze(out), cmap='jet', vmin=0., vmax=1.)
    plt.colorbar()
    plt.show()

    print('conv_map')