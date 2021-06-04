import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.Metric import Dice

from ControlGroup.AddClinicalFeature.withoutDisLastfc.ModelwithoutDis import ResNeXt

from SSHProject.BasicTool.MeDIT.Statistics import BinaryClassification
from SSHProject.BasicTool.MeDIT.Others import IterateCase


def ModelTest(weights_list=None, data_type=None):
    if data_type is None:
        data_type = ['alltrain', 'test']
    from Metric.classification_statistics import get_auc, draw_roc
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    model_folder = model_root + '/ResNeXt_CBAM_CV_20201117_five_lf'
    bc = BinaryClassification()

    fpr_list, tpr_list, auc_list = [], [], []
    for type in data_type:
        spliter = DataSpliter()
        sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(type))

        if type == 'test':
            data = DataManager(sub_list=sub_list)
            data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
            data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
            data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
            data.AddOne(Feature(data_root + '/FiveClinicalbGS.csv'), is_input=True)
            data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        else:
            data = DataManager(sub_list=sub_list)
            data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
            data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
            data.AddOne(Image2D(data_root + '/DwiSlice/', shape=input_shape))
            data.AddOne(Feature(data_root + '/FiveClinicalbGS.csv'), is_input=True)
            data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
        cv_pred_list, cv_label_list = [], []

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

            pred_list, label_list = [], []
            model.eval()
            for inputs, outputs in data_loader:
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(*inputs)[:, 1]
                pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
                label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())

            cv_pred_list.append(pred_list)
            cv_label_list.append(label_list)

            del model, weights_path

        cv_pred = np.array(cv_pred_list)
        cv_label = np.array(cv_label_list)
        mean_pred = np.mean(cv_pred, axis=0)
        mean_label = np.mean(cv_label, axis=0)
        np.save(
            os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/AddClinical/FiveFeatureLFbGS',
                         type + '_JSPH_nodis&fivefeature_label.npy'),
            mean_label)
        np.save(
            os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/AddClinical/FiveFeatureLFbGS',
                         type + '_JSPH_nodis&fivefeature_pred.npy'),
            mean_pred)

        fpr, tpr, auc = get_auc(mean_pred, mean_label)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)

        print(type)
        bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())

    draw_roc(fpr_list, tpr_list, auc_list, name_list=['alltrian', 'test'])


def ModelSUH(weights_list=None):
    from Metric.classification_statistics import get_auc, draw_roc
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    model_folder = model_root + '/ResNeXt_CBAM_CV_20201117_five_lf'
    bc = BinaryClassification()

    fpr_list, tpr_list, auc_list = [], [], []

    data = DataManager()
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
    data.AddOne(Feature(data_root + '/FiveClinical.csv'), is_input=True)
    data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Positive'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []

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

        pred_list, label_list = [], []
        model.eval()
        for inputs, outputs in data_loader:
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(*inputs)[:, 1]
            if isinstance((1 - preds).cpu().data.numpy().squeeze().tolist(), float):
                pred_list.append((preds).cpu().data.numpy().squeeze().tolist())
                label_list.append((outputs).cpu().data.numpy().astype(int).squeeze().tolist())
            else:
                pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
                label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    np.save(
        os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/AddClinical/FiveFeatureLFbGS',
                     'SUH_nodis&fivefeature_label.npy'),
        mean_label)
    np.save(
        os.path.join(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/AddClinical/FiveFeatureLFbGS',
                     'SUH_nodis&fivefeaturepred.npy'),
        mean_pred)

    fpr, tpr, auc = get_auc(mean_pred, mean_label)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc)

    result1 = bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())
    print(result1)


if __name__ == '__main__':
    # model_root = r'X:\Temp'
    # data_root = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide'

    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
    # data_root = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500'

    # EnsembleInference('test')
    # EnsembleInference('alltrain')

    ModelTest()
    # ModelSUH()
    # ROCofModels(data_type=['alltrain'])

