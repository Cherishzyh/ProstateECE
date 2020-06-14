import os
import cv2
import matplotlib.pyplot as plt
import shutil
import csv
import numpy as np
from scipy import ndimage

from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01
from MeDIT.SaveAndLoad import LoadNiiData

from FilePath import process_folder


def GetECETrue(ECE):
    ECE_True = []
    for index in range(len(ECE)):
        if ECE[index] == 'True':
            ECE_True.append(index)
        else:
            continue
    return ECE_True


def GetPiradsMax(PIRADS):
    pirads_max = []
    PIRADS_sort = sorted(PIRADS)
    pirads_max.append(PIRADS.index(PIRADS_sort[-1]))
    return pirads_max


def GetVolumn(image):
    volumn = np.sum(image)
    return volumn


def SelectTwo(PIRADS, ECE, roi0_path, roi1_path):
    if ECE[0] != ECE[1]:
        roi = GetECETrue(ECE)
        # print(case, 'ECE=True:', roi)
    else:
        if PIRADS[0] != PIRADS[1]:
            roi = GetPiradsMax(PIRADS)
            # print(case, 'PIRADSMAX', roi)
        else:
            _, data0, _ = LoadNiiData(roi0_path)
            _, data1, _ = LoadNiiData(roi1_path)
            volumn0 = np.sum(data0)
            volumn1 = np.sum(data1)
            if volumn0 > volumn1:
                roi = [0]
            else:
                roi = [1]
            # print(case, 'VolumnMax', roi)
    return roi


def SelectThree(PIRADS, ECE, roi0_path, roi1_path, roi2_path):
    if ECE[0] == ECE[1] == ECE[2]:
        if PIRADS[0] == PIRADS[1] == PIRADS[2]:
            _, data0, _ = LoadNiiData(roi0_path)
            _, data1, _ = LoadNiiData(roi1_path)
            _, data2, _ = LoadNiiData(roi2_path)
            volumn = [np.sum(data0), np.sum(data1), np.sum(data2)]
            roi = [volumn.index(max(volumn[0], volumn[1], volumn[2]))]
            # print(case, 'VolumnMax', roi)
        else:
            roi = GetPiradsMax(PIRADS)
            # print(case, 'MaxPIRADS', roi)
    else:
        roi = GetECETrue(ECE)
        # print(case, 'ECETrue', roi)
    return roi


def CopyData(original_path, des_path):
    shutil.copy(original_path, des_path)


def SelectRoi():
    for case in sorted(os.listdir(process_folder)):
        case_folder = os.path.join(process_folder, case)
        csv_path = os.path.join(case_folder, 'roi.csv')
        roi0_path = os.path.join(case_folder, 'roi0.nii')
        roi1_path = os.path.join(case_folder, 'roi1.nii')
        roi2_path = os.path.join(case_folder, 'roi2.nii')
        roi_path_list = [roi0_path, roi1_path, roi1_path]

        if os.path.exists(roi0_path):
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                PIRADS = [row['PIRADS'] for row in reader]
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                ECE = [row['ECE'] for row in reader]

            if len(PIRADS) == len(ECE) == 2:
                roi = SelectTwo(PIRADS, ECE, roi0_path, roi1_path)

                # print(case, roi)
            elif len(PIRADS) == len(ECE) == 3:
                roi = SelectThree(PIRADS, ECE, roi0_path, roi1_path, roi2_path)
                # print(case, roi)
            else:
                print(case, len(ECE))

            try:
                if len(roi) == 1:
                    des_path = os.path.join(case_folder, 'roi.nii')
                    CopyData(roi_path_list[roi[0]], des_path)
                else:
                    print(case)
            except Exception as e:
                print(case)


def MultiRoi():
    from MeDIT.SaveAndLoad import LoadNiiData
    data_folder = r'X:\PrcoessedData\ProstateCancerECE'
    multi_list = ['CHEN REN', 'CHEN ZHENG', 'DING YONG MING', 'DU KE BIN', 'FYK^fan yuan kai','GAO FA MING',
                  'GCD^gu chuan dao', 'GCF^gao chang fu','GENG LONG XIANG^GENG LONG XIANG','GSH^gao si hui','GU SI KANG',
                  'HFL^he fu lin', 'HONG QI YUN','JCG^jiang cheng guang','JIANG TIAN CAI^JIANG TIAN CAI','JJS^jin ju sheng',
                  'JZL^ji zhao lan', 'LCD^li cong de','LCZ^li chao zhi','LDJ^lei dao jin','LI JIA LIN','LLT^liang li tong',
                  'LRH^lu ronghua', 'MPX^ma pei xi','PENG XIAN XIANG','QB^qian bin','SBQ^shao ba qian','SHA GUANG YI^SHA GUANGYI',
                  'SHEN DAO XIANG', 'SSS^tang shan sheng','SUN BING LIANG','SUN QING ZHI','SUN ZHANG BAO','SXC^su xue cao',
                  'WANG YONG ZHENG', 'WEI CHANG HUA','WLB^wu lu bao','WLJ^wu liang ju ^13815870351^6924-31','WRM^weng rong ming',
                  'WU XIAO LEI', 'WZC^wu zhi chao','XJJ^xu jianjun','XJS^xu jingshan ^^6875-+04','XNB^xu neng bao','XSL^xu sen lou',
                  'XWC^xia wen cai', 'YANG YONG JIU','YFG^yang fu gang','YHX^yang hong xiao','YRE^yang rong r','YRF^yan rong fa',
                  'YU TU JUN', 'YYX^yin yong xing','ZGF^zhu guo fang','ZHAO YU LONG','ZHP^zhang heping','ZMJ^zhao mao jin',
                  'ZOU SHOU ZHONG', 'ZXJ^zhou xian jin ^^6698+5','ZXM^zhou xinmin','ZXT^zhu xiao ting','ZZF^zhou zheng fang ^^6698',
                  'ZZQ^zhou zu quan']

    print(len(multi_list))

    for a, case in enumerate(multi_list):
        data_path = os.path.join(data_folder, case)
        # if not os.path.exists(data_path):
        #     print(case)
        # else:
        #     data_list = os.listdir(data_path)
        #     roi_dict = {}
        #     for info in data_list:
        #         roi_dict['name'] = case
        #         if 'roi' in info and '.nii' in info:
        #             roi_dict[data_list.index(info)] = info
        #             roi_path = os.path.join(data_path, info)
        #             _, roi, _ = LoadNiiData(roi_path)
        #             label_im, nb_labels = ndimage.label(roi)
        #             for index in range(1, nb_labels+1):
        #                 num = str(label_im.tolist()).count(str(index))
        #                 if num >= 10:
        #                     roi_dict[index] = num
        #         else:
        #             continue
        #     print(a, roi_dict)
        roi_path = os.path.join(data_path, 'roi.nii')
        t2_path = os.path.join(data_path, 't2.nii')
        _, _, roi = LoadNiiData(roi_path)
        _, _, t2 = LoadNiiData(t2_path)
        label_im, nb_labels = ndimage.label(roi)
        Imshow3DArray(Normalize01(t2), roi=Normalize01(label_im))


def test():
    t2_path = r'X:\PrcoessedData\ProstateCancerECE\CHEN ZHENG\t2.nii'
    roi0_path = r'X:\PrcoessedData\ProstateCancerECE\CHEN ZHENG\roi.nii'
    # roi1_path = r'X:\PrcoessedData\ProstateCancerECE\SXC^su xue cao\roi1.nii'
    # roi2_path = r'X:\PrcoessedData\ProstateCancerECE\SXC^su xue cao\roi.nii'
    _, _, t2 = LoadNiiData(t2_path)
    _, _, roi0 = LoadNiiData(roi0_path)
    # _, roi1, _ = LoadNiiData(roi1_path)
    # _, roi2, _ = LoadNiiData(roi2_path)
    label_im0, nb_labels0 = ndimage.label(roi0)

    # label_im1, nb_labels1 = ndimage.label(roi1)
    # label_im2, nb_labels2 = ndimage.label(roi2)
    # roi0_list = []
    # roi1_list = []
    # roi2_list = []
    # for index in range(1, nb_labels0 + 1):
        # num = str(label_im0.tolist()).count(str(index))
        # roi0_list.append(num)
    new_mask1 = np.zeros(roi0.shape)
    new_mask2 = np.zeros(roi0.shape)
    # new_mask3 = np.zeros(roi0.shape)
    new_mask1[label_im0 == 1] = 1
    new_mask2[label_im0 == 2] = 1
    # new_mask3[label_im0 == 3] = 1
    # for index in range(1, nb_labels1 + 1):
    #     num = str(label_im1.tolist()).count(str(index))
    #     roi1_list.append(num)
    # for index in range(1, nb_labels2 + 1):
    #     num = str(label_im2.tolist()).count(str(index))
    #     roi2_list.append(num)

    # volumn = [np.sum(data0), np.sum(data1), np.sum(data2)]
    # roi = volumn.index(max(volumn[0], volumn[1], volumn[2]))
    # print(roi0_list, roi1_list, roi2_list)
    Imshow3DArray(Normalize01(t2), roi=[Normalize01(new_mask1), Normalize01(new_mask2)])


def ShowMuliROI():
    data_folder = r'X:\PrcoessedData\ProstateCancerECE'
    case_list = os.listdir(data_folder)
    for case in case_list:
        case_path = os.path.join(data_folder, case)
        if os.path.exists(os.path.join(case_path, 'roi0.nii')):
            print(case)

if __name__ == '__main__':
    # SelectRoi()
    # for case in sorted(os.listdir(process_folder)):
    #     case_folder = os.path.join(process_folder, case)
    #     roi_path = os.path.join(case_folder, 'adc.nii')
    #     if not os.path.exists(roi_path):
    #         print(case)
    # test()
    # MultiRoi()
    ShowMuliROI()

