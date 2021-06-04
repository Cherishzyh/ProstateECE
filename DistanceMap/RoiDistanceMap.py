from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation, binary_closing
import os

from MeDIT.SaveAndLoad import LoadImage
# from SSHProject.BasicTool.MeDIT.Visualization import ShowColorByRoi
# from SSHProject.BasicTool.MeDIT.ArrayProcess import ExtractPatch
from BasicTool.MeDIT.Visualization import ShowColorByRoi
from BasicTool.MeDIT.ArrayProcess import ExtractPatch, ExtractBlock
from ECEDataProcess.DataProcess.MaxRoi import GetRoiCenter, GetRoiCenterBefore


def BlurryEdge(roi, step=10):
    kernel = np.ones((3, 3))
    result = np.zeros_like(roi).astype(float)
    temp_roi = deepcopy(roi)
    for index in range(step):
        result += 1. / step * temp_roi.astype(float)
        temp_roi = binary_dilation(temp_roi, kernel)

    return result


def ExtractEdge(roi, kernel=np.ones((3, 3))):
    # plt.subplot(121)
    # plt.imshow(roi[0, ...], cmap='gray')
    # plt.subplot(122)
    # plt.imshow(binary_dilation(roi[0, ...].astype(bool), kernel, iterations=2).astype(int), cmap='gray')
    # plt.show()
    #
    # plt.subplot(121)
    # plt.imshow(roi[0, ...], cmap='gray')
    # plt.subplot(122)
    # plt.imshow(binary_erosion(roi[0, ...].astype(bool), kernel, iterations=2).astype(int), cmap='gray')
    # plt.show()
    #
    # plt.imshow(binary_dilation(roi[0, ...].astype(bool), kernel, iterations=2).astype(int) - \
    #        binary_erosion(roi[0, ...].astype(bool), kernel, iterations=2).astype(int), cmap='gray')
    # plt.show()
    return binary_dilation(roi.astype(bool), kernel, iterations=2).astype(int) - \
           binary_erosion(roi.astype(bool), kernel, iterations=2).astype(int)


def DetectRegion(roi0, roi1):
    kernel = np.ones((3, 3))
    roi0_edge = ExtractEdge(roi0, kernel)
    roi1_edge = ExtractEdge(roi1, kernel)

    roi1_out = roi1 - roi0
    roi1_out[roi1_out < 0] = 0
    roi1_out[roi1_out == 255] = 0


    region = roi1_out + (roi1_edge * roi0_edge)
    region[region > 1] = 1
    return region


def DetectCloseRegion(roi0, roi1, step=10, kernel=np.ones((3, 3))):
    roi0_edge, roi1_edge = ExtractEdge(roi0), ExtractEdge(roi1)
    assert((roi0_edge * roi1_edge).sum() == 0)

    diff = np.zeros_like(roi0)
    ratio = 0.
    for index in range(step):
        diff = roi0_edge * roi1_edge
        if diff.sum() > 0:
            ratio = (step - index) / step
            break
        roi0_edge = binary_dilation(roi0_edge.astype(bool), kernel)
        roi1_edge = binary_dilation(roi1_edge.astype(bool), kernel)

    return diff, ratio


def FindRegion(roi0, roi1):
    # 寻找结合点
    step = 10
    region = DetectRegion(roi0, roi1)
    if region.sum() >= 1:
        blurry = BlurryEdge(region, step)
    else:
        diff, ratio = DetectCloseRegion(roi0, roi1)
        blurry = ratio * BlurryEdge(diff.astype(int), step)

    # plt.subplot(121)
    # plt.imshow(roi0 + roi1)
    # plt.colorbar()
    # plt.subplot(122)
    # plt.imshow(blurry, vmax=1., vmin=0.)
    # plt.colorbar()
    # plt.show()

    return blurry


def Test():
    roi = np.zeros((256, 256))
    roi1 = np.zeros((256, 256))
    roi2 = np.zeros((256, 256))

    roi[50:150, 50:150] = 1
    roi1[60:100, 60:100] = 1
    roi2[40:100, 40:100] = 1

    # blurry = FindRegion(roi, roi1)
    blurry = FindRegion(roi, roi2)


def MergedDistanceMap():

    data_folder = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy'

    roi_folder = os.path.join(data_folder, 'cancer_slice')
    gland_folder = os.path.join(data_folder, 'kindey_slice')
    ct_folder = os.path.join(data_folder, 'ct_slice')

    for idx, case in enumerate(os.listdir(ct_folder)):
        case = '601548_-_slice8.npy'
        case_name = case[: case.index('_-_')]
        ct = np.squeeze(np.load(os.path.join(ct_folder, case)))
        roi = np.squeeze(np.load(os.path.join(roi_folder, case)))
        gland = np.squeeze(np.load(os.path.join(gland_folder, case)))

        gland_close = binary_closing(gland, iterations=3)

        blurry = FindRegion(gland_close, roi)
        blurry_1 = FindRegion(gland, roi)

        blurry_roi = np.where(blurry < 0.1, 0, 1)
        merged_roi = ShowColorByRoi(ct, blurry, blurry_roi, color_map='jet', is_show=False)

        plt.subplot(131)
        plt.axis('off')
        plt.imshow(ct, cmap='gray')
        plt.contour(roi, colors='r')
        plt.contour(gland, colors='y')
        plt.subplot(132)
        plt.axis('off')
        plt.imshow(merged_roi, cmap='jet')
        plt.subplot(133)
        plt.axis('off')
        plt.imshow(blurry_1, cmap='jet')
        # plt.show()
        plt.savefig(os.path.join(r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/atten_image', '{}.jpg'.format(case_name)))
        plt.close()
        np.save(os.path.join(r'/home/zhangyihong/Documents/Kindey901/Kindey_npy/atten_slice', case), blurry)
        # plt.suptitle(case_name)
        break

    # fig = plt.figure(figsize=(9, 6))
    #
    # l = 0.92
    # b = 0.12
    # w = 0.015
    # h = 0.37
    # # 对应 l,b,w,h；设置colorbar位置；
    # rect = [l, b, w, h]
    # cbar_ax = fig.add_axes(rect)
    #
    # plt.subplot(2, 3, 1)
    # plt.axis('off')
    # plt.imshow(t2_list[0], cmap='gray', )
    # plt.contour(prostate_list[0], colors='r', linewidths=0.5)
    # plt.contour(pcas_list[0], colors='y', linewidths=0.5)
    #
    # plt.subplot(2, 3, 4)
    # plt.imshow(dismap_list[0], cmap='jet')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 2)
    # plt.axis('off')
    # plt.imshow(t2_list[1], cmap='gray')
    # plt.contour(prostate_list[1], colors='r', linewidths=0.5)
    # plt.contour(pcas_list[1], colors='y', linewidths=0.5)
    #
    # plt.subplot(2, 3, 5)
    # plt.imshow(dismap_list[1], cmap='jet')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 3)
    # plt.axis('off')
    # plt.imshow(t2_list[2], cmap='gray')
    # plt.contour(prostate_list[2], colors='r', linewidths=0.5)
    # plt.contour(pcas_list[2], colors='y', linewidths=0.5)
    #
    # plt.subplot(2, 3, 6)
    # plt.imshow(dismap_list[2], cmap='jet')
    # plt.colorbar(cax=cbar_ax)
    # plt.axis('off')
    #
    # plt.gca().set_axis_off()
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #
    # plt.subplots_adjust(left=None, bottom=None, right=0.9, top=None,
    #                     wspace=0.01, hspace=0.01)

    # fig.subplots_adjust(right=0.9)

    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(save_path + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    # plt.close()
    # plt.clf()


def MergedDistanceMapTest():

    data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'

    roi_folder = os.path.join(data_folder, 'RoiSlice')
    gland_folder = os.path.join(data_folder, 'ProstateSlice')
    ct_folder = os.path.join(data_folder, 'T2Slice')

    for idx, case in enumerate(os.listdir(ct_folder)):
        ct = np.squeeze(np.load(os.path.join(ct_folder, case)))
        roi = np.squeeze(np.load(os.path.join(roi_folder, case)))
        gland = np.squeeze(np.load(os.path.join(gland_folder, case)))

        # gland_close = binary_closing(gland, iterations=7)

        case_name = case[:case.index('_slice')]
        blurry = FindRegion(gland, roi)
        blurry_1 = FindRegion(roi, gland)


        # blurry_roi = np.where(blurry < 0.1, 0, 1)
        # merged_roi = ShowColorByRoi(ct, blurry, blurry_roi, color_map='jet', is_show=False)

        plt.subplot(121)
        plt.imshow(blurry, cmap='jet')
        plt.subplot(122)
        plt.imshow(blurry_1, cmap='jet')
        # plt.contour(roi, colors='r')
        # plt.contour(gland, colors='y')
        plt.show()
        # np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/DistanceMap/Validation', case), blurry)
        # plt.suptitle(case_name)

    # fig = plt.figure(figsize=(9, 6))
    #
    # l = 0.92
    # b = 0.12
    # w = 0.015
    # h = 0.37
    # # 对应 l,b,w,h；设置colorbar位置；
    # rect = [l, b, w, h]
    # cbar_ax = fig.add_axes(rect)
    #
    # plt.subplot(2, 3, 1)
    # plt.axis('off')
    # plt.imshow(t2_list[0], cmap='gray', )
    # plt.contour(prostate_list[0], colors='r', linewidths=0.5)
    # plt.contour(pcas_list[0], colors='y', linewidths=0.5)
    #
    # plt.subplot(2, 3, 4)
    # plt.imshow(dismap_list[0], cmap='jet')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 2)
    # plt.axis('off')
    # plt.imshow(t2_list[1], cmap='gray')
    # plt.contour(prostate_list[1], colors='r', linewidths=0.5)
    # plt.contour(pcas_list[1], colors='y', linewidths=0.5)
    #
    # plt.subplot(2, 3, 5)
    # plt.imshow(dismap_list[1], cmap='jet')
    # plt.axis('off')
    #
    # plt.subplot(2, 3, 3)
    # plt.axis('off')
    # plt.imshow(t2_list[2], cmap='gray')
    # plt.contour(prostate_list[2], colors='r', linewidths=0.5)
    # plt.contour(pcas_list[2], colors='y', linewidths=0.5)
    #
    # plt.subplot(2, 3, 6)
    # plt.imshow(dismap_list[2], cmap='jet')
    # plt.colorbar(cax=cbar_ax)
    # plt.axis('off')
    #
    # plt.gca().set_axis_off()
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #
    # plt.subplots_adjust(left=None, bottom=None, right=0.9, top=None,
    #                     wspace=0.01, hspace=0.01)

    # fig.subplots_adjust(right=0.9)

    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(save_path + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    # plt.close()
    # plt.clf()


def ShowCase():
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import ConnectionPatch
    pca_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\PCaSlice'
    prostate_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ProstateSlice'
    t2_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\T2Slice'

    case = 'CHEN JIA ZHEN_-_slice12.npy'
    t2 = np.squeeze(np.load(os.path.join(t2_folder, case)))
    prostate = np.squeeze(np.load(os.path.join(prostate_folder, case)))
    pca = np.squeeze(np.load(os.path.join(pca_folder, case)))

    center = GetRoiCenterBefore(pca)

    t2_crop_100, _ = ExtractPatch(t2, (100, 100), center_point=center)
    pca_crop_100, _ = ExtractPatch(pca, (100, 100), center_point=center)
    prostate_crop_100, _ = ExtractPatch(prostate, (100, 100), center_point=center)

    dis_map_100 = FindRegion(prostate_crop_100, pca_crop_100)

    center = GetRoiCenterBefore(pca_crop_100)
    center = (center[0] - 10, center[1] + 10)
    t2_crop, _ = ExtractPatch(t2_crop_100, (15, 15), center_point=center)
    pca_crop, _ = ExtractPatch(pca_crop_100, (15, 15), center_point=center)
    prostate_crop, _ = ExtractPatch(prostate_crop_100, (15, 15), center_point=center)
    dis_map_crop, _ = ExtractPatch(dis_map_100, (15, 15), center_point=center)

    from MeDIT.Visualization import MergeImageWithRoi
    from MeDIT.Normalize import Normalize01

    fig, ax = plt.subplots()
    plt.axis('off')
    # plt.imshow(np.squeeze(t2_crop_100), cmap='gray')
    # plt.contour(np.squeeze(prostate_crop_100), colors='r', linewidths=1)
    # plt.contour(np.squeeze(pca_crop_100), colors='y', linewidths=1)
    plt.imshow(np.squeeze(dis_map_100), cmap='jet')

    axins2 = ax.inset_axes([1.1, 0, 1, 1])
    axins2.set_ylim(center[0]-7.5, center[0]+7.5)
    axins2.set_xlim(center[1]-7.5, center[1]+7.5)
    axins2.set_yticks([])
    axins2.set_xticks([])

    # merge = MergeImageWithRoi(Normalize01(np.squeeze(t2_crop)), roi=[
    #     np.squeeze(prostate_crop), np.squeeze(pca_crop)
    # ])
    # axins = ax.inset_axes([1.1, 0, 1, 1])
    # axins.imshow(merge[:-1, 1:, :])
    # axins.grid(True, 'major', 'both', lw=.5)
    # my_ticks = np.arange(-0.5, 13.5, 1)

    # （x0, y0, width, height）
    axins = ax.inset_axes([1.1, 0, 1, 1])
    axins.imshow(np.squeeze(dis_map_crop), cmap = 'jet')
    # axins.imshow(np.squeeze(t2_crop), cmap='gray')
    # axins.contour(np.squeeze(prostate_crop), colors='r', linewidths=.5)
    # axins.contour(np.squeeze(pca_crop), colors='y', linewidths=.5)
    axins.grid(True, 'major', 'both', lw=.5)
    my_ticks = np.arange(-0.5, 14.5, 1)

    axins.set_ylim()
    axins.set_yticks(my_ticks)
    axins.set_yticklabels([])

    axins.set_xlim()
    axins.set_xticks(my_ticks)
    axins.set_xticklabels(([]))

    mark_inset(ax, axins2, loc1=2, loc2=3, fc="none", ec='k', lw=1)
    fig.tight_layout()
    # plt.savefig(r'C:\Users\ZhangYihong\Desktop\ECE_dis.jpg', dpi=600)
    #
    plt.show()
    plt.close()
    ################################################################################

    # fig, ax = plt.subplots()
    # plt.imshow(np.squeeze(dis_map_crop), cmap='jet')
    # plt.colorbar()
    # ax.set_ylim()
    # ax.set_yticks(my_ticks)
    # ax.set_yticklabels([])
    #
    # ax.set_xlim()
    # ax.set_xticks(my_ticks)
    # ax.set_xticklabels(([]))
    #
    # plt.grid()
    # fig.tight_layout()
    # plt.savefig(r'C:\Users\ZhangYihong\Desktop\dis.jpg', dpi=600)
    # plt.show()


def SaveAllFigure():
    #######################################################################################################

    distance_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\DistanceMap'
    t2_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\T2Slice'
    pca_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\PCaSlice'
    pro_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ProstateSlice'

    for case in os.listdir(distance_folder):
        attention_map = np.squeeze(np.load(os.path.join(distance_folder, case)))
        t2 = np.squeeze(np.load(os.path.join(t2_folder, case)))
        pca = np.squeeze(np.load(os.path.join(pca_folder, case)))
        pro = np.squeeze(np.load(os.path.join(pro_folder, case)))
        # attention_map[attention_map > 0] = 1
        print(case)
        plt.suptitle(case)
        plt.subplot(121)
        plt.imshow(t2, cmap='gray')
        plt.contour(pca)
        plt.contour(pro)
        plt.subplot(122)
        plt.imshow(attention_map, cmap='jet')
        plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop\DIS', '{}.jpg'.format(case)))
        plt.close()

    #######################################################################################################


def SaveMultiScale():
    import torch
    import torch.nn.functional as F

    attention_map = np.squeeze(
        np.load(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\DistanceMap\CHEN JIA ZHEN_-_slice12.npy'))
    atten_crop, _ = ExtractPatch(attention_map, (192, 192))

    atten_crop = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(atten_crop), dim=0), dim=0)

    atten_crop_96 = F.interpolate(atten_crop, size=(96, 96), mode='bilinear', align_corners=True)
    atten_crop_48 = F.interpolate(atten_crop, size=(48, 48), mode='bilinear', align_corners=True)
    atten_crop_24 = F.interpolate(atten_crop, size=(24, 24), mode='bilinear', align_corners=True)
    atten_crop_12 = F.interpolate(atten_crop, size=(12, 12), mode='bilinear', align_corners=True)

    plt.imshow(np.squeeze(atten_crop.numpy()), cmap='jet')
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', '{}_192.jpg'.format('CHEN JIA ZHEN_-_slice12')),
                format='jpg',
                dpi=600, bbox_inches='tight', pad_inches=0.00)

    plt.imshow(np.squeeze(atten_crop_96.numpy()), cmap='jet')
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', '{}_96.jpg'.format('CHEN JIA ZHEN_-_slice12')),
                format='jpg',
                dpi=600, bbox_inches='tight', pad_inches=0.00)

    plt.imshow(np.squeeze(atten_crop_48.numpy()), cmap='jet')
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', '{}_48.jpg'.format('CHEN JIA ZHEN_-_slice12')),
                format='jpg',
                dpi=600, bbox_inches='tight', pad_inches=0.00)

    plt.imshow(np.squeeze(atten_crop_24.numpy()), cmap='jet')
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', '{}_24.jpg'.format('CHEN JIA ZHEN_-_slice12')),
                format='jpg',
                dpi=600, bbox_inches='tight', pad_inches=0.00)

    plt.imshow(np.squeeze(atten_crop_12.numpy()), cmap='jet')
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', '{}_12.jpg'.format('CHEN JIA ZHEN_-_slice12')),
                format='jpg',
                dpi=600, bbox_inches='tight', pad_inches=0.00)

    plt.show()


def QAQ():
    from BasicTool.MeDIT.ArrayProcess import ExtractPatch

    case_folder = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide'
    t2_path = os.path.join(case_folder, 'T2Slice\CHEN JIA ZHEN_-_slice12.npy')
    adc_path = os.path.join(case_folder, 'AdcSlice\CHEN JIA ZHEN_-_slice12.npy')
    dwi_path = os.path.join(case_folder, 'DwiSlice\CHEN JIA ZHEN_-_slice12.npy')
    pro_path = os.path.join(case_folder, 'ProstateSlice\CHEN JIA ZHEN_-_slice12.npy')
    pca_path = os.path.join(case_folder, 'PCaSlice\CHEN JIA ZHEN_-_slice12.npy')
    atten_path = os.path.join(case_folder, 'DistanceMap\CHEN JIA ZHEN_-_slice12.npy')

    path_list = [t2_path, adc_path, dwi_path, pro_path, pca_path, atten_path]
    name_list = ['t2', 'adc', 'dwi', 'pro', 'pca', 'atten']
    # for index, path in enumerate(path_list):
    #     if index == 1:
    #         break
        # data = np.squeeze(np.load(path))
        # pro = np.squeeze(np.load(pro_path))
        # pca = np.squeeze(np.load(pca_path))
        # name = name_list[index]
        #
        # shape = (180, 180)
        # center = GetRoiCenterBefore(pro)
        # data, _ = ExtractPatch(data, patch_size=shape, center_point=center)
        # pro, _ = ExtractPatch(pro, patch_size=shape, center_point=center)
        # pca, _ = ExtractPatch(pca, patch_size=shape, center_point=center)
        #
        # plt.imshow(data, cmap='gray')
        # plt.contour(pro, colors='r')
        # plt.contour(pca, colors='y')
        # plt.gca().set_axis_off()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', '{}.jpg'.format(name)),
        #             format='jpg',
        #             dpi=600, bbox_inches='tight', pad_inches=0.00)

    cancer_roi = np.squeeze(np.load(pca_path))
    prostate_roi = np.squeeze(np.load(pro_path))
    t2 = np.squeeze(np.load(t2_path))
    blurry = FindRegion(prostate_roi, cancer_roi)

    # crop
    center = GetRoiCenterBefore(prostate_roi)
    t2_crop, _ = ExtractPatch(t2, (180, 180), center_point=center)
    cancer_crop, _ = ExtractPatch(cancer_roi, (180, 180), center_point=center)
    prostate_crop, _ = ExtractPatch(prostate_roi, (180, 180), center_point=center)
    blurry_crop, _ = ExtractPatch(blurry, (180, 180), center_point=center)

    blurry_roi = np.where(blurry_crop < 0.1, 0, 1)
    merged_roi = ShowColorByRoi(t2_crop, blurry_crop, blurry_roi, color_map='jet', is_show=False)
    plt.imshow(merged_roi, cmap='jet')
    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop', 'atten.jpg'),
                format='jpg',
                dpi=600, bbox_inches='tight', pad_inches=0.00)


if __name__ == '__main__':
    MergedDistanceMap()








