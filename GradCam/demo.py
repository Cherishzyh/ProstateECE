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
from GradCam.grad_cam_main import get_device, get_classtable, load_images, save_gradcam


def demo_my(model, input_list, input_class, output_dir):
    """
    Generate Grad-GradCam at different layers of ResNet-152
    """
    # device = get_device(cuda)
    # Synset words
    # classes = get_classtable()
    # input_class

    # MyModel
    # model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = torch.argmax(input_class)  # "bull mastif"


    # Images
    # images, raw_images = load_images(image_paths)
    # inputs
    t2_images = input_list[0]
    images = torch.cat(input_list, dim=1).float()

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    # print(torch.argmax(input_class), probs)
    # ids_ = torch.LongTensor([[target_class]]*len(images)).to(device)
    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, target_class, float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet50", target_layer, input_class[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=t2_images, # images[j]
            )


def demo2(model, image_paths, output_dir, cuda):
    """
    Generate Grad-GradCam at different layers of ResNet-152
    """

    device = get_device(cuda)

    # Synset words
    classes = get_classtable()

    # MyModel
    # model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # The four residual layers
    target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_class = 243  # "bull mastif"

    # Images
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer)

        for j in range(len(images)):
            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[ids == target_class])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, "resnet152", target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=raw_images[j],
            )


if __name__ == '__main__':
    from DataSet.MyDataLoader import LoadTVData, LoadTestData
    from MyModel.ResNet50 import ResNet, Bottleneck

    device = torch.device('cpu')
    model_path = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNet505InputOneHot/checkpoint.pt'
    train_loader, valid_loader = LoadTVData(r'/home/zhangyihong/Documents/ProstateECE/NPYOnehot', shape=(192, 192), is_test=True)
    test_loader = LoadTestData(r'/home/zhangyihong/Documents/ProstateECE/NPYOnehot', shape=(192, 192))
    output_dir = r'/home/zhangyihong/Documents/ProstateECE/grad_cam'

    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.load(model_path))

    for i, (inputs, outputs) in enumerate(train_loader):
        t2, dwi, adc, prostate, roi = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
        ece = outputs[0].to(device)

        input_list = [t2.to(device), dwi.to(device), adc.to(device), prostate.to(device), roi.to(device)]

        # pca_out, pirads_out = model(input_list, roi.to(device))
        if ece == 0:
            input_class = torch.tensor([1, 0]).long()
        else:
            input_class = torch.tensor([0, 1]).long()

        demo_my(model, input_list, input_class.to(device), output_dir)



