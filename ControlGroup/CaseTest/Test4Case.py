import SimpleITK as sitk
import torch

from CnnTools.T4T.Utility.Data import *
from BasicTool.MeDIT.Statistics import BinaryClassification
from BasicTool.MeDIT.Others import IterateCase
from BasicTool.MeDIT.SaveAndLoad import LoadImage

from SYECE.model import ResNeXt
# from SYECE.ModelWithoutDis import ResNeXt
from ECEDataProcess.DataProcess.MaxRoi import GetRoiCenter
from DistanceMap.RoiDistanceMap import FindRegion, ExtractEdge



def LoadData(data_folder):
    # t2_path = os.path.join(data_folder, 't2.nii')
    # dwi_path = os.path.join(data_folder, 'dwi_Reg.nii')
    # adc_path = os.path.join(data_folder, 'adc_Reg.nii')
    # prostate_path = os.path.join(data_folder, 'ProstateROI_TrumpetNet.nii.gz')
    # pca_path = os.path.join(data_folder, 'roi.nii')

    t2_path = os.path.join(data_folder, 't2_5x5.nii')
    dwi_path = os.path.join(data_folder, 'dwi_Reg.nii')
    adc_path = os.path.join(data_folder, 'adc_Reg.nii')
    prostate_path = os.path.join(data_folder, 'prostate_roi_5x5.nii.gz')
    pca_path = os.path.join(data_folder, 'pca_roi_5x5.nii.gz')

    _, t2, _ = LoadImage(t2_path, dtype=np.float32)
    _, dwi, _ = LoadImage(dwi_path, dtype=np.float32)
    _, adc, _ = LoadImage(adc_path, dtype=np.float32)
    _, prostate, _ = LoadImage(prostate_path, dtype=np.float32)
    _, pca, _ = LoadImage(pca_path, dtype=np.float32)

    return t2.transpose((2, 0, 1)), dwi.transpose((2, 0, 1)), adc.transpose((2, 0, 1)), \
           prostate.transpose((2, 0, 1)), pca.transpose((2, 0, 1))


def GetROISlice(roi_data):
    PCa_slice = []
    for slice in range(roi_data.shape[0]):
        if np.sum(roi_data[slice, ...]) == 0:
            continue
        else:
            PCa_slice.append(slice)
    return PCa_slice


def DatabySlice(data, slice):
    return data[slice, ...]


def CropData(data, crop_shape, center, is_roi=False):
    from MeDIT.ArrayProcess import ExtractPatch

    # Normalization
    if not is_roi:
        data -= np.mean(data)
        data /= np.std(data)

    # Crop
    data_crop, _ = ExtractPatch(data, crop_shape, center_point=center)

    return data_crop
    # return data


def Process(data, slice, center, is_roi=False):

    data_slice = DatabySlice(data, slice)

    crop_shape = (192, 192)
    # crop_shape = (280, 280)

    data_slice_crop = CropData(data_slice, crop_shape, center, is_roi)

    return data_slice_crop


def Run(case_folder):
    input_list = []
    t2, dwi, adc, prostate, pca = LoadData(case_folder)

    slice_list_pca = GetROISlice(pca)
    slice_list_pro = GetROISlice(prostate)
    slice_list = [slice for slice in slice_list_pca if slice in slice_list_pro]
    # print(slice_list)

    for slice in slice_list:
        data_slice_list = []
        center, _ = GetRoiCenter(pca[slice, ...])
        data_slice_list.append(Process(t2, slice, center))
        data_slice_list.append(Process(adc, slice, center))
        data_slice_list.append(Process(dwi, slice, center))
        data_slice_list.append(Process(prostate, slice, center, is_roi=True))
        data_slice_list.append(Process(pca, slice, center, is_roi=True))
        input_list.append(data_slice_list)

    return input_list


def ModelTest(data_folder, model_folder, case_name, weights_list=None):
    # label_df = pd.read_csv(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ece.csv', index_col='case')
    label_df = pd.read_csv(r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/label.csv', index_col='case')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    label_list, case_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXt(3, 2).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
                                     one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        all_case_pred_list = []
        model.eval()
        for case in case_name:
            case_all_slice_list = Run(os.path.join(data_folder, case[: case.index('_-_slice')]))
            # case_all_slice_list = Run(os.path.join(data_folder, case))
            all_slice_preds_list = []

            if cv_index == 0:
                # label_list.append((label_df.loc[case])['ece'])
                label_list.append((label_df.loc[case[: case.index('_-_slice')]])['label'])
                case_list.append(case[: case.index('_-_slice')])

            print('in cv {}, predict {}'.format(cv_index, case))
            # predict for each slice
            for case_one_slice_list in case_all_slice_list:
                distance_map = FindRegion(case_one_slice_list[3], case_one_slice_list[4]) # attention map
                distance_map = np.where(distance_map >= 0.1, 1, 0).astype(np.float32) # binary attention map

                # distance_map = ExtractEdge(np.squeeze(case_one_slice_list[3]), kernel=np.ones((7, 7))).astype(np.float32) # prostate boundary

                # distance_map = case_one_slice_list[4]  # pca roi

                t2, dwi, adc = case_one_slice_list[0], case_one_slice_list[1], case_one_slice_list[2]

                inputs_list = MoveTensorsToDevice([torch.tensor(t2[np.newaxis, np.newaxis, ...]),
                                                  torch.tensor(dwi[np.newaxis, np.newaxis, ...]),
                                                  torch.tensor(adc[np.newaxis, np.newaxis, ...]),
                                                  torch.tensor(distance_map[np.newaxis, np.newaxis, ...])],
                                                  device)

                # prediction for a slice
                preds = model(*inputs_list)[:, 1]

                # prediction_list for all slice
                all_slice_preds_list.append((preds).cpu().data.numpy().squeeze().tolist())

            # prediction_list for all cases for a cv-train
            all_case_pred_list.append(all_slice_preds_list)

        # prediction_list for all cases for  all cv-train
        cv_pred_list.append(all_case_pred_list)

        del model, weights_path

    mean_pred = []
    for index in range(len(cv_pred_list[0])):
        # the average prediction of cv-train for all cases
        mean_pred.append(np.mean(np.array([list[index] for list in cv_pred_list]), axis=0).tolist())

    return case_list, mean_pred, label_list


def ComputeConfusionMetric(data_path, YI=0.5):
    from BasicTool.MeDIT.SaveAndLoad import LoadH5

    case_list = os.listdir(data_path)
    pred_list, label_list = [], []
    TP, TN, FP, FN = 0, 0, 0, 0
    for case in case_list:
        # case = 'BAO ZHENG LI.h5'
        case_path = os.path.join(data_path, case)
        slice_preds, label = LoadH5(case_path, tag=['prediction', 'label'], data_type=[np.float32, np.uint8])
        print(slice_preds.tolist())
        slice_preds_list = np.where(slice_preds > YI, 1., 0.).tolist()
        if 1. in slice_preds_list:
            if label == 1:
                TP += 1
            else:
                FP += 1
        else:
            if label == 1:
                FN += 1
            else:
                TN += 1
    print(TP, TN, FP, FN)
    return TP, TN, FP, FN


def ComputeAUC(data_path):
    from BasicTool.MeDIT.SaveAndLoad import LoadH5

    case_list = os.listdir(data_path)
    pred_list, label_list = [], []
    for case in case_list:
        case_path = os.path.join(data_path, case)
        slice_preds, label = LoadH5(case_path, tag=['prediction', 'label'], data_type=[np.float32, np.uint8])
        slice_preds = [1 - pred for pred in slice_preds]

        pred_list.append(np.max(np.array(slice_preds)))
        label_list.append(label.astype(int).tolist())
    from BasicTool.MeDIT.Statistics import BinaryClassification
    bc = BinaryClassification()
    bc.Run(pred_list, label_list)



if __name__ == '__main__':
    from BasicTool.MeDIT.SaveAndLoad import SaveH5

    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20201130_BinaryAttenMap'
    # data_root = r'/home/zhangyihong/Documents/ProstateECE/ResampleData'
    data_root = r'/home/zhangyihong/Documents/ProstateECE/ProstateCancerECE_SUH'

    ##################################TRAIN#######################################################
    # train_list = sorted(os.listdir(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice'))
    # train_list.remove('Test')
    # train_name = [name[:name.index('.npy')] for name in train_list]
    #
    # case_list, mean_pred, label_list = ModelTest(data_root, model_root, train_name, weights_list=None)
    # for index, case in enumerate(case_list):
    #     save_path = os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/BinaryAtten/Train', case+'.h5')
    #     SaveH5(save_path, data=[mean_pred[index], label_list[index]],
    #            tag=['prediction', 'label'], data_type=[np.float32, np.uint8])

    ##################################TEST#######################################################
    # test_list = sorted(os.listdir(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice/Test'))
    # test_name = [name[:name.index('.npy')] for name in test_list]
    # case_list, mean_pred, label_list = ModelTest(data_root, model_root, test_name, weights_list=None)
    #
    # for index, case in enumerate(case_list):
    #     save_path = os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/BinaryAtten/Test', case+'.h5')
    #     SaveH5(save_path, data=[mean_pred[index], label_list[index]],
    #            tag=['prediction', 'label'], data_type=[np.float32, np.uint8])


    ##################################SUH#######################################################
    SUH_list = sorted(os.listdir(r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500/AdcSlice'))
    SUH_name = [name[:name.index('.npy')] for name in SUH_list]
    case_list, mean_pred, label_list = ModelTest(data_root, model_root, SUH_name, weights_list=None)

    for index, case in enumerate(case_list):
        save_path = os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/CaseH5/BinaryAtten/SUH', case+'.h5')
        SaveH5(save_path, data=[mean_pred[index], label_list[index]],
               tag=['prediction', 'label'], data_type=[np.float32, np.uint8])


    # ComputeAUC(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\SUH')
    # print('O.O')







