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
    # target_layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    target_layers = ["layer4"]
    # target_class = torch.argmax(input_class)  # "bull mastif"
    target_class = input_class


    # Images
    # images, raw_images = load_images(image_paths)
    # inputs
    t2_images = input_list[0]
    images = torch.cat(input_list, dim=1).float().unsqueeze(dim=0)
    # images = images.unsqueeze()

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    # print(torch.argmax(input_class))
    # ids_ = torch.LongTensor([[target_class]]*len(images)).to(device)
    ids_ = torch.tensor([[target_class]] * 1).long().to(device)
    gcam.backward(ids=ids_)

    for target_layer in target_layers:
        print("Generating Grad-GradCam @{}".format(target_layer))

        # Grad-GradCam
        regions = gcam.generate(target_layer=target_layer, target_shape=(224, 224))

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
                        j, "resnet50", target_layer, target_class
                    ),
                ),
                gcam=regions[j, 0],
                raw_image=t2_images, # images[j]
            )


if __name__ == '__main__':
    device = torch.device('cpu')

    model = models.resnet50(pretrained=True)

    image_path = r'cat_dog_original.jpg'


    images, raw_images = load_images(image_path)

    classes = get_classtable()

    # 243, 281, 282

    demo_my(model, images, 243, r'')

    # demo2([''], 'GradCamFolder', 'cpu')



