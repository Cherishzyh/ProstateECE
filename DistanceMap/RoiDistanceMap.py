from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import os

from MeDIT.SaveAndLoad import LoadImage


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
    from SSHProject.BasicTool.MeDIT.Visualization import ShowColorByRoi
    from SSHProject.BasicTool.MeDIT.ArrayProcess import ExtractPatch
    from ECEDataProcess.DataProcess.MaxRoi import GetRoiCenter
    data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
    cancer_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/RoiSlice'
    prostate_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ProstateSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/T2Slice'

    cancer_folder = os.path.join(cancer_folder, 'Test')
    # cancer_list = os.listdir(cancer_folder)
    prostate_folder = os.path.join(prostate_folder, 'Test')
    t2_folder = os.path.join(t2_folder, 'Test')
    # prostate_list = os.listdir(prostate_folder)
    cancer_list = ['CHEN REN_slice11.npy', 'CHEN JIA ZHEN_slice12.npy', 'Fang zhi hua_slice7.npy']
    save_path = os.path.join(r'/home/zhangyihong/Documents/ProstateECE/DistanceMapImage', 'DistanceMap')
    t2_list, dismap_list = [], []
    prostate_list, pcas_list = [], []
    for idx, case in enumerate(cancer_list):
        case_name = case[:case.index('.npy')]
        cancer_roi = np.squeeze(np.load(os.path.join(cancer_folder, case)))
        prostate_roi = np.squeeze(np.load(os.path.join(prostate_folder, case)))
        t2 = np.squeeze(np.load(os.path.join(t2_folder, case)))
        blurry = FindRegion(prostate_roi, cancer_roi)

        # crop
        center = GetRoiCenter(prostate_roi)
        t2_crop, _ = ExtractPatch(t2, (120, 120), center_point=center)
        cancer_crop, _ = ExtractPatch(cancer_roi, (120, 120), center_point=center)
        prostate_crop, _ = ExtractPatch(prostate_roi, (120, 120), center_point=center)
        blurry_crop, _ = ExtractPatch(blurry, (120, 120), center_point=center)

        blurry_roi = np.where(blurry_crop < 0.1, 0, 1)
        merged_roi = ShowColorByRoi(t2_crop, blurry_crop, blurry_roi, color_map='jet', is_show=False)
        prostate_list.append(prostate_crop)
        pcas_list.append(cancer_crop)
        t2_list.append(t2_crop)
        dismap_list.append(merged_roi)
        # blurry = blurry[np.newaxis, ...]
        # np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/DistanceMap/Validation', case), blurry)
        # plt.suptitle(case_name)

    fig = plt.figure(figsize=(9, 6))

    l = 0.92
    b = 0.12
    w = 0.015
    h = 0.37
    # 对应 l,b,w,h；设置colorbar位置；
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)

    plt.subplot(2, 3, 1)
    plt.axis('off')
    plt.imshow(t2_list[0], cmap='gray', )
    plt.contour(prostate_list[0], colors='r', linewidths=0.5)
    plt.contour(pcas_list[0], colors='y', linewidths=0.5)

    plt.subplot(2, 3, 4)
    plt.imshow(dismap_list[0], cmap='jet')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.axis('off')
    plt.imshow(t2_list[1], cmap='gray')
    plt.contour(prostate_list[1], colors='r', linewidths=0.5)
    plt.contour(pcas_list[1], colors='y', linewidths=0.5)

    plt.subplot(2, 3, 5)
    plt.imshow(dismap_list[1], cmap='jet')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.axis('off')
    plt.imshow(t2_list[2], cmap='gray')
    plt.contour(prostate_list[2], colors='r', linewidths=0.5)
    plt.contour(pcas_list[2], colors='y', linewidths=0.5)

    plt.subplot(2, 3, 6)
    plt.imshow(dismap_list[2], cmap='jet')
    plt.colorbar(cax=cbar_ax)
    plt.axis('off')

    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(left=None, bottom=None, right=0.9, top=None,
                        wspace=0.01, hspace=0.01)

    # fig.subplots_adjust(right=0.9)

    # plt.savefig(save_path + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    # plt.savefig(save_path + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    plt.close()
    plt.clf()


if __name__ == '__main__':
    # Test()
    # data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ProstateSlice/Test'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/T2Slice/Test'
    # save_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/BoundarySlice/Test'
    data_folder = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/ProstateSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/T2Slice'
    save_path = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/BoundarySlice'
    for case in os.listdir(data_folder):
        # if case == 'Test':
        #     continue
        # else:
        prostate = np.load(os.path.join(data_folder, case))
        boundary = ExtractEdge(np.squeeze(prostate), kernel=np.ones((7, 7)))
        # np.save(os.path.join(save_path, case), boundary[np.newaxis])

        # t2 = np.load(os.path.join(t2_folder, case))
        # plt.subplot(221)
        # plt.imshow(np.squeeze(t2), cmap='gray')
        # plt.contour(np.squeeze(prostate), colors='r')
        # plt.subplot(222)
        # plt.imshow(np.squeeze(t2), cmap='gray')
        # plt.contour(ExtractEdge(np.squeeze(prostate), kernel=np.ones((3, 3))), colors='r')
        # plt.subplot(223)
        # plt.imshow(np.squeeze(t2), cmap='gray')
        # plt.contour(ExtractEdge(np.squeeze(prostate), kernel=np.ones((5, 5))), colors='r')
        # plt.subplot(224)
        # plt.imshow(np.squeeze(t2), cmap='gray')
        # plt.contour(ExtractEdge(np.squeeze(prostate), kernel=np.ones((7, 7))), colors='r')
        # plt.show()


