import h5py
import numpy as np
import os
import matplotlib.pyplot as plt


def GetData(data_folder):
    file_list = os.listdir(data_folder)
    image_list = []
    label_list = []

    for file in file_list:
        file_path = os.path.join(data_folder, file)

        # data read
        with h5py.File(file_path, 'r') as h5_file:
            image = np.asarray(h5_file['input_0'], dtype=np.float32)
            label = np.asarray(h5_file['output_0'], dtype=np.uint8)

        image_list.append(image)
        label_list.append(label)

    return np.asarray(image_list), np.asarray(label_list)


def GeneratorData(data_folder, batch_size):
    file_list = os.listdir(data_folder)
    image_list = []
    label_list = []

    while True:
        for file in file_list:
            file_path = os.path.join(data_folder, file)

            # data read
            with h5py.File(file_path, 'r') as h5_file:
                image = np.asarray(h5_file['input_0'], dtype=np.float32)
                label = np.asarray(h5_file['output_0'], dtype=np.uint8)

            image_list.append(image)
            label_list.append(label)

            if len(image_list) >= batch_size:
                yield np.asarray(image_list), np.asarray(label_list)
                image_list = []
                label_list = []


def main():
    data_folder = r'X:\CNNFormatData\ProstateCancerECE\AllData\Validation'
    file_list = os.listdir(data_folder)

    for file in file_list:
        file_path = os.path.join(data_folder, file)

        # data read
        with h5py.File(file_path, 'r') as h5_file:
            image = np.asarray(h5_file['input_0'], dtype=np.float32)
            label = np.asarray(h5_file['output_0'], dtype=np.uint8)

        plt.imshow(image[0, ...], cmap='gray')
        plt.contour(label[0, ...], colors='r')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()