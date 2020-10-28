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
from SSHProject.BasicTool.MeDIT.Visualization import Imshow3DArray
from SSHProject.BasicTool.MeDIT.Normalize import Normalize01


def demo_my(model, input_list, input_class):
    """
    Generate Grad-GradCam at different layers of ResNet-152
    """

    device = torch.device('cpu')

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