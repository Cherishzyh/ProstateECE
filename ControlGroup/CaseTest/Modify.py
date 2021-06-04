######################################################
# 看max pred 和 max roi 的层是否能对应
# 理论上讲，max pred >= max roi

# if max pred is 1 and max roi is 1:
#     pass
# elif max pred is 1 and max roi is 0:
#     the slice of max roi is non-ece,
#     update the data
# elif max pred is 0 and max roi is 1:
#     wrong pred and In theory shouldn't exist
# else max pred is 0 and max roi is 0:
#     pass

# if max pred slice is max roi slice:
#   pass
# else
#
import numpy as np
import matplotlib.pyplot as plt

from ControlGroup.CaseTest.Test4Case import *
from SSHProject.BasicTool.MeDIT.SaveAndLoad import LoadH5
from SSHProject.BasicTool.MeDIT.Visualization import Imshow3DArray
from SSHProject.BasicTool.MeDIT.Normalize import Normalize01
from ECEDataProcess.DataProcess.MaxRoi import KeepLargest


def demo():
    pred_list, label_list = [], []
    for case in sorted(os.listdir(adc_folder)):
        # print(case)
        # get max pred slice
        case_path = os.path.join(H5_path, case[: case.index('_slice')]+'.h5')
        slice_preds, label = LoadH5(case_path, tag=['prediction', 'label'], data_type=[np.float32, np.uint8])
        slice_preds = [1 - pred for pred in slice_preds]
        index = slice_preds.index(max(slice_preds))

        # get data and slice
        t2, dwi, adc, prostate, pca = LoadData(os.path.join(data_root, case[: case.index('_slice')]))
        _, _, pca = KeepLargest(pca)
        slice_list_pca = GetROISlice(pca)
        slice_list_pro = GetROISlice(prostate)
        slice_list = [slice for slice in slice_list_pca if slice in slice_list_pro]

        # for slice in slice_list:
        #     center, _ = GetRoiCenter(pca[slice, ...])
        #     t2_slice_3d = Process(t2, slice, center)
        #     adc_slice_3d = Process(adc, slice, center)
        #     dwi_slice_3d = Process(dwi, slice, center)
        #     pro_slice_3d = Process(prostate, slice, center, is_roi=True)
        #     pca_slice_3d = Process(pca, slice, center, is_roi=True)
        #     # data_slice_list.append(np.array(feature_list))

        # get npy slice
        npy_slice = int(case[case.index('_slice')+6: case.index('.npy')])
        t2_slice = np.load(os.path.join(t2_folder, case))
        adc_slice = np.load(os.path.join(adc_folder, case))
        dwi_slice = np.load(os.path.join(dwi_folder, case))
        pro_slice = np.load(os.path.join(pro_folder, case))
        pca_slice = np.load(os.path.join(pca_folder, case))

        center = GetRoiCenterBefore(pca[npy_slice, ...])
        t2_slice_3d = Process(t2, npy_slice, center)
        adc_slice_3d = Process(adc, npy_slice, center)
        dwi_slice_3d = Process(dwi, npy_slice, center)
        pro_slice_3d = Process(prostate, npy_slice, center, is_roi=True)
        pca_slice_3d = Process(pca, npy_slice, center, is_roi=True)

        plt.subplot(131)
        plt.imshow(np.squeeze(t2[npy_slice, ...]), cmap='gray')
        plt.contour(np.squeeze(pca[npy_slice, ...]), colors='r')
        plt.scatter(center[1], center[0], marker='o')
        plt.subplot(132)
        plt.imshow(np.squeeze(t2_slice), cmap='gray')
        plt.scatter(140, 140, marker='o')
        plt.subplot(133)
        plt.imshow(np.squeeze(t2_slice_3d), cmap='gray')
        plt.scatter(140, 140, marker='o')
        plt.show()


        # print('{}: max pred slice {}, npy slice {}'.format(case, slice_list[index], npy_slice))
        # compare

        # if npy_slice == slice_list[index]:
        #     print('{}: max pred slice {}, npy slice {}'.format(case, slice_list[index], npy_slice))
        # else:
            # if max pred == 1 but max roi == 0
            # 保存最大层

            # continue


def GetUseSliceNum(case, folder):
    case_list = sorted(os.listdir(folder))
    if 'Test' or 'DSR^dai shou rong_slice16.npy' in case_list:
        case_list.remove('Test')
        case_list.remove('DSR^dai shou rong_slice16.npy')
    if "_-_slice" in case_list[0]:
        case_name = [case[: case.index('_-_slice')] for case in sorted(case_list)]
    else:
        case_name = [case[: case.index('_slice')] for case in sorted(case_list)]

    if case in case_name:
        index = case_name.index(case)
        case_slice = case_list[index]
        slice_index = int(case_slice[case_slice.index('_slice')+6: case_slice.index('.npy')])
    else:
        slice_index = -1
    return slice_index


def Get3DPred(case, folder, if_negative=False):
    case_path = os.path.join(folder, case+'.h5')
    slice_preds, label = LoadH5(case_path, tag=['prediction', 'label'], data_type=[np.float32, np.uint8])
    if if_negative:
        slice_preds = [1 - pred for pred in slice_preds]
    return slice_preds, label


def Get2DPred(case, csv_path):
    # slice_pred = r'/home/zhangyihong/Documents/ProstateECE/Result/PAGNet_test.csv'
    slice_df = pd.read_csv(csv_path, index_col='case')
    pred_2d = slice_df.loc[case]['Pred']
    label_2d = slice_df.loc[case]['Label']
    return pred_2d, label_2d


def GetSliceList(case, folder, is_show=False):
    t2, _, _, prostate, pca = LoadData(os.path.join(folder, case))
    if is_show:
        Imshow3DArray(Normalize01(t2.transpose(1, 2, 0)),
                      roi=[Normalize01(prostate.transpose(1, 2, 0)), Normalize01(pca.transpose(1, 2, 0))])
    _, _, pca = KeepLargest(pca)
    slice_list_pca = GetROISlice(pca)
    slice_list_pro = GetROISlice(prostate)
    slice_list = [slice for slice in slice_list_pca if slice in slice_list_pro]
    return slice_list


def Compare(case):
    # get 2d
    slice_pred = r'/home/zhangyihong/Documents/ProstateECE/Result/PAGNet_suh.csv'
    pred_2d, label_2d = Get2DPred(case, slice_pred)

    # get 3d
    preds, label_3d = Get3DPred(case, H5_path, if_negative=True)
    slice_list = GetSliceList(case, data_root)

    # slice_num
    slice_index = GetUseSliceNum(case, adc_folder)

    index = slice_list.index(slice_index)
    pred_3d = preds[index]

    if label_2d == label_3d:
        pass
    else:
        raise Exception

    if abs(pred_2d - pred_3d) < 1e-6:
        print('*')
        # print('{}: preds are the same'.format(case))
    else:
        print('{}: preds are different'.format(case))


def Correct(case):
    '''
    label == 1,
    one_slice_pred == 0,
    multi_slice_pred == 1,
    use max slice pred take the place of the max roi slice

    ['CDM^chang deng ming', 'CDX^cao da xin', 'GWJ^guan wei jun', 'JBS^jin bao song ^^6698-48', 'TENG DE HONG',
     'TNS^tang nian shun', 'TWZ^tong wei zhi', 'WKZ^wang kai zhong', 'WXP^wu xu ping', 'WZC^wei zheng cang',
     'XIE NING SHENG^XIE NING SHENG', 'XNY^xiao nai yu', 'ZHANG ZE REN', 'ZSN^zhang si niu', 'ZYG^zhu yuan gui']
    '''
    # get 2d
    pred_2d, label_2d = Get2DPred(case, r'/home/zhangyihong/Documents/ProstateECE/Result/PAGNet_train.csv')

    # get 3d
    preds, label_3d = Get3DPred(case, H5_path, if_negative=True)

    return pred_2d, max(preds), label_2d

    # if label_2d == label_3d == 1:
    #     if pred_2d <= 0.800000739 and max(preds) > 0.7999990579999999:
    #         # use max slice pred take the place of the max roi slice
    #         print(case)
            # pass


if __name__ == '__main__':

    # model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200820'
    data_root = r'/home/zhangyihong/Documents/ProstateECE/ResampleData'
    # data_root = r'/home/zhangyihong/Documents/ProstateECE/ProstateCancerECE_SUH'

    H5_path = r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/PAGNet_RightCrop/Train'
    adc_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice'
    # adc_folder = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/T2Slice/Test'
    dwi_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/DwiSlice/Test'
    pro_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ProstateSlice/Test'
    pca_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/RoiSlice/Test'


    #########################draw the different hist between max roi and max pred#######################################
    # difference_list = []
    # for case in sorted(os.listdir(H5_path)):
    #     case = case[: case.index('.h5')]
    #     pred_2d, pred_3d, _ = Correct(case)
    #     difference_list.append(abs(pred_3d - pred_2d))
    #
    # plt.hist(difference_list, bins=20)
    # plt.show()


    ################ the number of cases that the max roi slice is different from the max pred slice ###################
    # count = 0
    #
    # for case in sorted(os.listdir(H5_path)):
    #     case = case[: case.index('.h5')]
    #     slice_max_roi = GetUseSliceNum(case, adc_folder)
    #     slice_list = GetSliceList(case, data_root, is_show=False)
    #     slice_pred, _ = Get3DPred(case, H5_path, if_negative=False)
    #     slice_max_pred = slice_list[slice_pred.tolist().index(max(slice_pred))]
    #
    #

    # Compare('SUN QI MEI^SUN QI MEI^^5072-11')
    # Compare('WANG YE LIANG')


    ################ show the cases that 2d pred is different from 3d pred in the same slice############################
    # case_list = ['ZHANG ZHEN GUO', 'ZHANG ZHEN']
    # case_list = ['XIANG SI XIAN', 'SUN QI MEI^SUN QI MEI^^5072-11', 'WANG YE LIANG']
    # folder = r'X:\StoreFormatData\ProstateCancerECE\ResampleData'
    # from BasicTool.MeDIT.Visualization import Imshow3DArray
    # from BasicTool.MeDIT.Normalize import Normalize01
    # # for case in sorted(case_list):
    # t2, dwi, adc, prostate, pca = LoadData(os.path.join(folder, 'ZHANG ZHEN GUO'))
    # Imshow3DArray(Normalize01(t2.transpose(1, 2, 0)), roi=[Normalize01(prostate.transpose(1, 2, 0)), Normalize01(pca.transpose(1, 2, 0))])


    ##################################### write csv for right crop result #########################################
    # pred_2d_list, pred_3d_list, label_list = [], [], []
    # for case in sorted(os.listdir(r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/PAGNet_RightCrop/Test')):
    #     # Compare(case[: case.index('.h5')])
    #     pred_2d, pred_3d, label = Correct(case[: case.index('.h5')])
    #     pred_2d_list.append(pred_2d)
    #     pred_3d_list.append(pred_3d)
    #     label_list.append(label)
    # df = pd.DataFrame({'case': sorted(os.listdir(r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/PAGNet_RightCrop/Test')),
    #                    'label': label_list, 'pred one slice': pred_2d_list, 'pred multi slice': pred_3d_list})
    # df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/PAGNet_RightCrop/test_compare.csv', index=False)













