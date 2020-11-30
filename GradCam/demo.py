from __future__ import print_function

import os.path as osp

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from GradCam.grad_cam import GradCAM
from GradCam.grad_cam_main import save_gradcam

device = torch.device('cpu')

def demo_my(model, input_list, input_class):
    """
    Generate Grad-GradCam at different layers of ResNet-152
    """
    model.eval()

    target_layers = ["layer4"]
    target_class = torch.argmax(input_class)

    gcam = GradCAM(model=model)
    probs = gcam.forward(input_list)

    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer, target_shape=(192, 192))

        print("\t#{} ({:.5f})".format(target_class, float(probs[:, 1])))

        gradcam = save_gradcam(
            gcam=regions[0, 0, ...],
        )

    return probs[:, 1], gradcam


if __name__ == '__main__':
    from SSHProject.CnnTools.T4T.Utility.Data import *
    from SSHProject.BasicTool.MeDIT.Normalize import Normalize01
    from SSHProject.BasicTool.MeDIT.Visualization import FusionImage
    from SYECE.model import ResNeXt
    # from SYECE.ModelWithoutDis import ResNeXt
    from SSHProject.BasicTool.MeDIT.ArrayProcess import ExtractPatch
    from ECEDataProcess.DataProcess.MaxRoi import GetRoiCenter
    from torch.utils.data import DataLoader

    device = torch.device('cpu')
    # model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200820/CV_0/31--5.778387.pt'
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200814/CV_1/154--7.698224.pt'
    data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
    output_dir = r'/home/zhangyihong/Documents/ProstateECE/grad_cam'

    model = ResNeXt(3, 2).to(device)
    model.load_state_dict(torch.load(model_root))
    model.to(device)

    input_shape = (192, 192)

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format('test'))

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DistanceMap/Test', shape=input_shape, is_roi=True))
    data.AddOne(Image2D(data_root + '/ProstateSlice/Test', shape=input_shape, is_roi=True))
    data.AddOne(Image2D(data_root + '/RoiSlice/Test', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
    # data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
    data_loader = DataLoader(data, batch_size=1, shuffle=False)

    for i, (inputs, outputs) in enumerate(data_loader):
        if i == 107 or i == 130:
            t2, adc, dwi, dis_map = inputs[0], inputs[1], inputs[2], inputs[3]
            ece = outputs[0].to(device)

            input_list = [t2.to(device), adc.to(device), dwi.to(device), dis_map.to(device)]
            # input_list = [t2.to(device), adc.to(device), dwi.to(device)]

            if ece == 0:
                input_class = torch.tensor([1, 0]).long()
            else:
                input_class = torch.tensor([0, 1]).long()

            prob, gradcam = demo_my(model, input_list, input_class.to(device))

            center, box = GetRoiCenter(np.squeeze(inputs[4].numpy()))

            center = (center[0], center[1])
            if i == 107:
                shape = (120, 120)
            else:
                shape = (160, 160)
            t2_cropped, _ = ExtractPatch(np.squeeze(inputs[0].numpy()), patch_size=shape, center_point=center)
            gradcam_cropped, _ = ExtractPatch(np.squeeze(gradcam), patch_size=shape, center_point=center)

            merged_image = FusionImage(Normalize01(np.squeeze(t2_cropped)),
                                       Normalize01(np.squeeze(gradcam_cropped)), is_show=False)

            plt.subplot(121)
            plt.axis('off')
            plt.imshow(np.squeeze(t2_cropped), cmap='gray')
            plt.subplot(122)
            plt.axis('off')
            plt.imshow(np.squeeze(merged_image), cmap='jet')

            plt.gca().set_axis_off()
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())

            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=0.00, hspace=0.01)

            # plt.savefig(r'/home/zhangyihong/Documents/ProstateECE/Paper/' + str(i) + '-RES.tif', format='tif', dpi=600, bbox_inches='tight', pad_inches=0.00)
            plt.show()
            plt.close()
            plt.clf()
        else:
            continue


