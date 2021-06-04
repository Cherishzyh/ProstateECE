import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from BasicTool.MeDIT.Visualization import Imshow3DArray
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.SaveAndLoad import LoadNiiData



def NPY():
    t2_case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\T2Slice\Train'
    prostate_case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\ProstateSlice\Train'
    roi_case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\RoiSlice\Train'
    case_list = os.listdir(t2_case_folder)
    for case in case_list:

        t2_path = os.path.join(t2_case_folder, case)
        prostate_path = os.path.join(prostate_case_folder, case)
        roi_path = os.path.join(roi_case_folder, case)

        t2_data = np.transpose(np.load(t2_path), [1, 2, 0])
        roi_data = np.transpose(np.load(roi_path), [1, 2, 0])
        prostate_data = np.transpose(np.load(prostate_path), [1, 2, 0])

        if len(list(np.unique(prostate_data))) == 1:
            print(case)
        # print(case, np.unique(roi_data))
        # plt.imshow(np.squeeze(t2_data), cmap='gray')
        # plt.contour(np.squeeze(prostate_data), colors='r')
        # plt.contour(np.squeeze(roi_data), colors='y')
        # plt.show()
# NPY()

def NII():
    case_folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    case_list = os.listdir(case_folder)
    case = 'FAN DA HAI'
    case_path = os.path.join(case_folder, case)
    t2_path = os.path.join(case_path, 't2.nii')
    prostate_path = os.path.join(case_path, 'ProstateROI_TrumpetNet.nii.gz')
    cancer_path = os.path.join(case_path, 'roi.nii')

    _, t2_data, _ = LoadNiiData(t2_path)
    _, prostate_data, _ = LoadNiiData(prostate_path)
    _, cancer_data, _ = LoadNiiData(cancer_path)

    Imshow3DArray(Normalize01(t2_data), roi=[Normalize01(prostate_data), Normalize01(cancer_data)])
# NII()

def Show():
    import torch
    from SYECE.model import ResNeXt

    from GradCam.demo import demo_my

    from BasicTool.MeDIT.ArrayProcess import ExtractPatch
    from BasicTool.MeDIT.Visualization import ShowColorByRoi
    from BasicTool.MeDIT.Visualization import FusionImage

    # data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
    # label_csv = os.path.join(data_folder, 'ece.csv')
    # label_df = pd.read_csv(label_csv, index_col='case')
    # t2_folder = os.path.join(data_folder, 'T2Slice/Test')
    # adc_folder = os.path.join(data_folder, 'AdcSlice/Test')
    # dwi_folder = os.path.join(data_folder, 'DwiSlice/Test')
    #
    # pro_folder = os.path.join(data_folder, 'ProstateSlice/Test')
    # pca_folder = os.path.join(data_folder, 'RoiSlice/Test')
    #
    # atten_folder = os.path.join(data_folder, 'DistanceMap/Test')

    data_folder = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500'
    label_csv = os.path.join(data_folder, 'label.csv')
    label_df = pd.read_csv(label_csv, index_col='case')

    t2_folder = os.path.join(data_folder, 'T2Slice')
    adc_folder = os.path.join(data_folder, 'AdcSlice')
    dwi_folder = os.path.join(data_folder, 'DwiSlice')

    pro_folder = os.path.join(data_folder, 'ProstateSlice')
    pca_folder = os.path.join(data_folder, 'PCaSlice')

    atten_folder = os.path.join(data_folder, 'DistanceMap')

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200814/CV_1/154--7.698224.pt'

    output_dir = r'/home/zhangyihong/Documents/ProstateECE/Image_external'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = ResNeXt(3, 2).to(device)
    model.load_state_dict(torch.load(model_root))
    model.to(device)

    input_shape = (192, 192)

    for index, case in enumerate(os.listdir(adc_folder)):
        t2_path = os.path.join(t2_folder, case)
        adc_path = os.path.join(adc_folder, case)
        dwi_path = os.path.join(dwi_folder, case)
        pro_path = os.path.join(pro_folder, case)
        pca_path = os.path.join(pca_folder, case)
        atten_path = os.path.join(atten_folder, case)


        t2 = np.squeeze(np.load(t2_path))
        adc = np.squeeze(np.load(adc_path))
        dwi = np.squeeze(np.load(dwi_path))
        pro = np.squeeze(np.load(pro_path))
        pca = np.squeeze(np.load(pca_path))
        atten = np.squeeze(np.load(atten_path))

        t2, _ = ExtractPatch(t2, input_shape)
        adc, _ = ExtractPatch(adc, input_shape)
        dwi, _ = ExtractPatch(dwi, input_shape)
        pro, _ = ExtractPatch(pro, input_shape)
        pca, _ = ExtractPatch(pca, input_shape)
        atten, _ = ExtractPatch(atten, input_shape)

        blurry_roi = np.where(atten < 0.1, 0, 1)
        merged_roi = ShowColorByRoi(t2, atten, blurry_roi, color_map='jet', is_show=False)

        # label = label_df.loc[case[:case.index('.npy')]]['ece']
        label = label_df.loc[case[:case.index('_-_slice')]]['label']
        # if label == 0:
        #     input_class = torch.tensor([1, 0]).long()
        # else:
        #     input_class = torch.tensor([0, 1]).long()
        t2_input = torch.from_numpy(t2[np.newaxis, np.newaxis, ...]).float()
        adc_input = torch.from_numpy(adc[np.newaxis, np.newaxis, ...]).float()
        dwi_input = torch.from_numpy(dwi[np.newaxis, np.newaxis, ...]).float()
        atten_input = torch.from_numpy(atten[np.newaxis, np.newaxis, ...]).float()


        input_list = [t2_input.to(device), adc_input.to(device), dwi_input.to(device), atten_input.to(device)]

        prob, gradcam_1 = demo_my(model, input_list, torch.tensor([1, 0]).long().to(device))
        _, gradcam_2 = demo_my(model, input_list, torch.tensor([0, 1]).long().to(device))

        merged_image_1 = FusionImage(Normalize01(np.squeeze(t2)),
                                   Normalize01(np.squeeze(gradcam_1)), is_show=False)
        merged_image_2 = FusionImage(Normalize01(np.squeeze(t2)),
                                   Normalize01(np.squeeze(gradcam_2)), is_show=False)

        plt.figure(figsize=(12, 8))

        plt.suptitle('case_name:{}, label:{}, pred: {:.2f}'.
                     format(case[:case.index('_-_')], label, float(1-prob.cpu().data.numpy())))

        plt.subplot(231)
        plt.axis('off')
        plt.imshow(t2, cmap='gray')
        plt.subplot(232)
        plt.axis('off')
        plt.imshow(adc, cmap='gray')
        plt.subplot(233)
        plt.axis('off')
        plt.imshow(t2, cmap='gray')
        plt.contour(pro, colors='r')
        plt.contour(pca, colors='y')

        plt.subplot(234)
        plt.axis('off')
        plt.imshow(merged_roi, cmap='jet')

        plt.subplot(235)
        plt.axis('off')
        plt.imshow(merged_image_1, cmap='jet')
        plt.subplot(236)
        plt.axis('off')
        plt.imshow(merged_image_2, cmap='jet')

        # plt.show()
        # plt.gca().set_axis_off()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(os.path.join(output_dir, '{}.jpg'.format(case[: case.index('_-_slice')])), format='jpg')
        plt.close()
Show()