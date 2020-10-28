from __future__ import print_function

import os.path as osp

# import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms




def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def load_images(image_paths):
    images = []
    raw_images = []
    print("Images:")
    if isinstance(image_paths, str):
        image, raw_image = preprocess(image_paths)
        images.append(image)
        raw_images.append(raw_image)
    else:
        for i, image_path in enumerate(image_paths):
            print("\t#{}: {}".format(i, image_path))
            image, raw_image = preprocess(image_path)
            images.append(image)
            raw_images.append(raw_image)
    return images, raw_images


def get_classtable():
    classes = []
    with open("synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)
    return classes


def preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))


def save_gradcam(gcam):
    gcam = gcam.cpu().numpy()

    ##################
    # if isinstance(raw_image, torch.Tensor):
    #     raw_image = torch.squeeze(raw_image).numpy()

    # plt.subplot(121)
    # plt.imshow(Normalize01(np.squeeze(gcam)), cmap='jet')
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(Normalize01(np.squeeze(raw_image)), cmap='jet')
    # plt.show()

    # merged_image = FusionImage(Normalize01(raw_image), Normalize01(gcam), is_show=is_show)
    return gcam



# def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
#     gcam = gcam.cpu().numpy()
#     cmap = cm.jet_r(gcam)[..., :3] * 255.0
#
#     if paper_cmap:
#         alpha = gcam[..., None]
#         gcam = alpha * cmap + (1 - alpha) * raw_image
#     else:
#         gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
#
#
#     ##################
#     if isinstance(raw_image, torch.Tensor):
#         raw_image = torch.squeeze(raw_image).numpy()
#
#     if paper_cmap:
#         alpha = gcam[..., None]
#         gcam = alpha * gcam + (1 - alpha) * raw_image
#     else:
#         gcam = (gcam.astype(np.float) + raw_image.astype(np.float)) / 2
#
#     cv2.imwrite(filename, np.uint8(gcam))


def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)


if __name__ == "__main__":
    # main()
    pass
