import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import torch
from torch.utils.data import DataLoader

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.BasicTool.MeDIT.Others import IterateCase
from Metric.classification_statistics import get_auc, draw_roc
# from Metric.MyMetric import BinaryClassification
from SSHProject.BasicTool.MeDIT.Statistics import BinaryClassification


def ModelJSPH(weights_list=None, is_dismap=True, data_type='test', store_path=r''):
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    bc = BinaryClassification(store_folder=store_path, store_format='eps')
    if is_dismap:
        from SYECE.model import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
    else:
        from SYECE.ModelWithoutDis import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200820'

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(data_type))

    if data_type == 'test':
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
        data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
        data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
        if is_dismap:
            data.AddOne(Image2D(data_root + '/DistanceMap/Test', shape=input_shape, is_roi=True))
            data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
        else:
            data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)

        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
        if is_dismap:
            data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
            data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
        else:
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
            if is_dismap:
                pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
                label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())

            else:
                pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
                label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())

        # if cv_index == 1:
        #     for num, pred in enumerate(pred_list):
        #         print(num, pred)
        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())

    return mean_pred, mean_label,


def ModelSUH(weights_list=None, is_dismap=True, store_path=r''):
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 1
    bc = BinaryClassification(store_folder=store_path, store_format='eps')
    if is_dismap:
        from SYECE.model import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
    else:
        from SYECE.ModelWithoutDis import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200820'

    data = DataManager()
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
    if is_dismap:
        data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
        data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Negative'), is_input=False)
    else:
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
                if is_dismap:
                    pred_list.append((1 - preds).cpu().data.numpy().squeeze().tolist())
                    label_list.append((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
                else:
                    pred_list.append((preds).cpu().data.numpy().squeeze().tolist())
                    label_list.append((outputs).cpu().data.numpy().astype(int).squeeze().tolist())

            else:
                if is_dismap:
                    pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
                    label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
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
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist(), r'/home/zhangyihong/Documents/ProstateECE/SUH_prob.eps')

    return mean_pred, mean_label


if __name__ == '__main__':
    model_pred, label = ModelJSPH(is_dismap=True, data_type='test', store_path=r'/home/zhangyihong/Documents/ProstateECE')
    # for num, pred in enumerate(model_pred):
    #     if 0.6 < pred < 0.9:
    #         print(num, pred)
    # print(model_pred[62], model_pred[57], model_pred[15], model_pred[38])
    # model_pred_nodis, label_nodis = ModelJSPH(is_dismap=False, data_type='test')
    # # print(label == label_nodis)
    # print(wilcoxon(model_pred, model_pred_nodis))

    # model_pred, label = ModelSUH(is_dismap=True)
    # model_pred_nodis, label_nodis = ModelSUH(is_dismap=False)
    # _, _ = ModelSUH(is_dismap=True)
    # _, _ = ModelSUH(is_dismap=False)
    # print(wilcoxon(model_pred, model_pred_nodis))



    # model_pred0, label0 = ModelJSPH(is_dismap=True, data_type='alltrain')
    # model_pred1, label1 = ModelJSPH(is_dismap=True, data_type='test')
    # model_pred2, label2 = ModelJSPH(is_dismap=False, data_type='alltrain')
    # model_pred3, label3 = ModelJSPH(is_dismap=False, data_type='test')
    # fpr0, tpr0, auc0 = get_auc(model_pred0, label0)
    # fpr1, tpr1, auc1 = get_auc(model_pred1, label1)
    # fpr2, tpr2, auc2 = get_auc(model_pred2, label2)
    # fpr3, tpr3, auc3 = get_auc(model_pred3, label3)
    # name_list = ['ModelP train', 'ModelP test', 'ModelT train', 'ModelT test']
    # plt.plot(fpr0, tpr0, color='tab:red', label=name_list[0] + ': ' + '%.3f'% auc0)
    # plt.plot(fpr1, tpr1, '--', color='tab:red', label=name_list[1] + ': ' + '%.3f' % auc1)
    # plt.plot(fpr2, tpr2, color='tab:blue', label=name_list[2] + ': ' + '%.3f' % auc2)
    # plt.plot(fpr3, tpr3, '--', color='tab:blue', label=name_list[3] + ': ' + '%.3f' % auc3)

    # model_pred0, label0 = ModelSUH(is_dismap=True)
    # model_pred1, label1 = ModelSUH(is_dismap=False)
    # fpr0, tpr0, auc0 = get_auc(model_pred0, label0)
    # fpr1, tpr1, auc1 = get_auc(model_pred1, label1)
    # name_list = ['ModelP', 'ModelT']
    # plt.plot(fpr0, tpr0, label=name_list[0] + ': ' + '%.3f' % auc0)
    # plt.plot(fpr1, tpr1, label=name_list[1] + ': ' + '%.3f' % auc1)
    # #
    # plt.plot([0, 1], [0, 1], '--', color='k')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc='lower right')
    # plt.show()