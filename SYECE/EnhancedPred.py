import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pandas as pd
import torch
from torch.utils.data import DataLoader

from MeDIT.Augment import *
from T4T.Utility.Data import *
from MeDIT.Others import IterateCase
from Metric.classification_statistics import get_auc, draw_roc

from Metric.MyMetric import BinaryClassification


param_config = {
    RotateTransform.name: {'theta': ['uniform', -10, 10]},
    ShiftTransform.name: {'horizontal_shift': ['uniform', -0.05, 0.05],
                          'vertical_shift': ['uniform', -0.05, 0.05]},
    ZoomTransform.name: {'horizontal_zoom': ['uniform', 0.95, 1.05],
                         'vertical_zoom': ['uniform', 0.95, 1.05]},
    FlipTransform.name: {'horizontal_flip': ['choice', True, False]},
    BiasTransform.name: {'center': ['uniform', -1., 1., 2],
                         'drop_ratio': ['uniform', 0., 1.]},
    NoiseTransform.name: {'noise_sigma': ['uniform', 0., 0.03]},
    ContrastTransform.name: {'factor': ['uniform', 0.8, 1.2]},
    GammaTransform.name: {'gamma': ['uniform', 0.8, 1.2]},
    ElasticTransform.name: ['elastic', 1, 0.1, 256]
}


def EnhancedTestSUH(is_dismap=True, param=None):
    data_root = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500'
    input_shape = (192, 192)
    batch_size = 2

    data = DataManager(augment_param=param)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
    if is_dismap:
        data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
        data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Negative'), is_input=False)
    else:
        data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Positive'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return data_loader


def ModelEnhancedSUH(weights_list=None, is_dismap=True):
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if is_dismap:
        from SYECE.model import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
    else:
        from SYECE.ModelWithoutDis import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200820'

    bc = BinaryClassification()

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

        pred_list_enhanced, label_list_enhanced = [], []
        model.eval()
        for i in range(9):
            print(i)
            pred_list, label_list = [], []
            if i == 0:
                data_loader = EnhancedTestSUH(is_dismap, param_config)
            else:
                data_loader = EnhancedTestSUH(is_dismap)

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
            pred_list_enhanced.append(pred_list), label_list_enhanced.append(label_list)

        cv_pred_list.append(np.mean(np.array(pred_list_enhanced), axis=0).tolist())
        cv_label_list.append(np.mean(np.array(label_list_enhanced), axis=0).tolist())

        fpr, tpr, auc = get_auc(np.mean(pred_list_enhanced, axis=0).tolist(),
                                np.mean(label_list_enhanced, axis=0).tolist())
        print('AUC: {}'.format(auc))
        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())
    return mean_pred, mean_label


def EnhancedTestJSPH(is_dismap=True, data_type='test', param=None):
    data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
    input_shape = (192, 192)
    batch_size = 2
    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(data_type))
    data = DataManager(sub_list=sub_list, augment_param=param)

    if data_type == 'test':
        data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
        data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
        data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
        if is_dismap:
            data.AddOne(Image2D(data_root + '/DistanceMap/Test', shape=input_shape, is_roi=True))
            data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
        else:
            data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
    else:
        data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
        if is_dismap:
            data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
            data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
        else:
            data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return data_loader


def ModelEnhancedJSPH(weights_list=None, is_dismap=True, data_type='test'):
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if is_dismap:
        from SYECE.model import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
    else:
        from SYECE.ModelWithoutDis import ResNeXt
        model_folder = model_root + '/ResNeXt_CBAM_CV_20200820'

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    bc = BinaryClassification()

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

        pred_list_enhanced, label_list_enhanced = [], []
        model.eval()
        for i in range(9):
            print(i)
            pred_list, label_list = [], []
            if i == 0:
                data_loader = EnhancedTestJSPH(is_dismap, data_type, param_config)
            else:
                data_loader = EnhancedTestJSPH(is_dismap, data_type)

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

            pred_list_enhanced.append(pred_list), label_list_enhanced.append(label_list)

        cv_pred_list.append(np.mean(pred_list_enhanced, axis=0).tolist())
        cv_label_list.append(np.mean(label_list_enhanced, axis=0).tolist())

        fpr, tpr, auc = get_auc(np.mean(pred_list_enhanced, axis=0).tolist(),
                                np.mean(label_list_enhanced, axis=0).tolist())
        print('AUC: {}'.format(auc))
        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())
    return mean_pred, mean_label



if __name__ == '__main__':
    from SYECE.ModelCompared import *
    from sklearn import metrics
    from scipy import stats

    # model_pred, label = ModelJSPH(is_dismap=True)
    # model_pred_nodis, label_nodis = ModelJSPH(is_dismap=False)
    # # print(wilcoxon(model_pred, model_pred_nodis))
    #
    # model_pred_enhanced, _ = ModelEnhancedJSPH(is_dismap=True)
    # model_pred_nodis_enhanced, _ = ModelEnhancedJSPH(is_dismap=False)
    #
    # # print(wilcoxon(model_pred, model_pred_nodis))
    # df = pd.DataFrame({'attention pred': model_pred, 'attention enhanced pred': model_pred_enhanced,
    #                    'no attention pred': model_pred_nodis, 'no attention enhanced pred': model_pred_nodis_enhanced,
    #                    'label': label})

    model_pred, label = ModelSUH(is_dismap=True)
    model_pred_nodis, label_nodis = ModelSUH(is_dismap=False)
    # print(wilcoxon(model_pred, model_pred_nodis))

    # model_pred_enhanced, _ = ModelEnhancedSUH(is_dismap=True)
    # model_pred_nodis_enhanced, _ = ModelEnhancedSUH(is_dismap=False)
    df = pd.DataFrame({'attention pred': model_pred, 'no attention pred': model_pred_nodis, 'label': label})
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/SUH_info.csv', mode='a+', header=False)

    # df = pd.read_csv(r'/home/zhangyihong/Documents/ProstateECE/info_JPSH.csv', index_col='num')
    # pred = []
    # label = []
    # for index in df.index:
    #     info = df.loc[index]
    #     pred.append((info['attention pred']))
    #     label.append(int(info['label']))
    # fpr, tpr, thresholds = metrics.roc_curve(label, pred)
    # auc = metrics.auc(fpr, tpr)
    # print(auc)
