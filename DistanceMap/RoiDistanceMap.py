from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation

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

    blurry = FindRegion(roi, roi1)
    blurry = FindRegion(roi, roi2)


if __name__ == '__main__':
    import numpy as np
    import os

    # cancer_roi = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\RoiSlice\Train\BI JUN_slice11.npy')
    # prostate_roi = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPY\ProstateSlice\Train\BI JUN_slice11.npy')
    # cancer_roi = np.squeeze(cancer_roi)
    # prostate_roi = np.squeeze(prostate_roi)
    # FindRegion(prostate_roi, cancer_roi)


    data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY/'
    cancer_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY/RoiSlice'
    prostate_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY/ProstateSlice'

    cancer_folder = os.path.join(cancer_folder, 'Validation')
    cancer_list = os.listdir(cancer_folder)
    prostate_folder = os.path.join(prostate_folder, 'Validation')
    # prostate_list = os.listdir(prostate_folder)

    for case in cancer_list:

        case_name = case[:case.index('.npy')]
        cancer_roi = np.squeeze(np.load(os.path.join(cancer_folder, case)))
        prostate_roi = np.squeeze(np.load(os.path.join(prostate_folder, case)))
        blurry = FindRegion(prostate_roi, cancer_roi)
        blurry = blurry[np.newaxis, ...]
        np.save(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/NPY/DistanceMap/Validation', case), blurry)
        # plt.imshow(blurry, vmax=1., vmin=0.)
        # plt.colorbar()
        # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/ProstateECE/NPY/DistanceMap', case_name+'.jpg'))
        # plt.close()

        print(case_name)
    # case_name = r'XWZ^xiong wei zhi_slice12.npy'
    # cancer_roi = np.squeeze(np.load(os.path.join(cancer_folder, case_name)))
    # prostate_roi = np.squeeze(np.load(os.path.join(prostate_folder, case_name)))
    # plt.suptitle(case_name)
    # plt.subplot(121)
    # plt.title('PCa')
    # plt.imshow(cancer_roi)
    # plt.subplot(122)
    # plt.title('Prostate')
    # plt.imshow(prostate_roi)
    # plt.show()
