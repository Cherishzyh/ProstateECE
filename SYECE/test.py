import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from T4T.Utility.Data import *

from SYECE.model import ResNeXt, ResNeXtOneInput
# from SYECE.ModelWithoutDis import ResNeXt

from MeDIT.Statistics import BinaryClassification
from MeDIT.Others import IterateCase
from MeDIT.ArrayProcess import ExtractBlock


def EnsembleInference(data_type, weights_list=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 20
    model_folder = model_root + '/PAGNet0429'
    bc = BinaryClassification()

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(data_type))

    # if data_type == 'test':
    #     data = DataManager()
    #     data.AddOne(Image2D(data_root + '/Test/T2Slice', shape=input_shape))
    #     data.AddOne(Image2D(data_root + '/Test/AdcSlice', shape=input_shape))
    #     data.AddOne(Image2D(data_root + '/Test/DwiSlice', shape=input_shape))
    #     data.AddOne(Image2D(data_root + '/Test/DistanceMap', shape=input_shape, is_roi=True))
    #     data.AddOne(Label(data_root + '/ece.csv'), is_input=False)
    #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    # else:
    #     data = DataManager()
    #     data.AddOne(Image2D(data_root + '/Train/T2Slice', shape=input_shape))
    #     data.AddOne(Image2D(data_root + '/Train/AdcSlice', shape=input_shape))
    #     data.AddOne(Image2D(data_root + '/Train/DwiSlice/', shape=input_shape))
    #     data.AddOne(Image2D(data_root + '/Train/DistanceMap', shape=input_shape, is_roi=True))
    #     data.AddOne(Label(data_root + '/ece.csv'), is_input=False)
    #     data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/ct_slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/atten_slice', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/label.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    cv_fcn_out_list = []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXtOneInput(1, 2).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        fcn_out_list = []
        model.eval()
        for inputs, outputs in data_loader:
            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            model_pred = model(*inputs)
            preds = model_pred[:, 1]
            # pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
            # label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
            # fcn_out_list.extend(model_pred[1].cpu().data.numpy().squeeze().tolist())

            pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
            label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())

        # bc.Run(pred_list, label_list)

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)
        # cv_fcn_out_list.append(fcn_out_list)

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    # cv_fcn = np.array(cv_fcn_out_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)
    # mean_fcn_out = np.mean(cv_fcn, axis=0)
    # if data_type == 'test':
    #     np.save(r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Result/PAGNet_test.npy', mean_pred)
    #     np.save(r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Result/PAGNet_test_label.npy', mean_label)
    # else:
    #     np.save(r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Result/PAGNet_train.npy', mean_pred)
    #     np.save(r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Result/PAGNet_label.npy', mean_label)
    # os.mkdir(r'/home/zhangyihong/Documents/Kindey901/Model/Result')
    if data_type == 'test':
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/test.npy', mean_pred)
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/test_label.npy', mean_label)
    else:
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/train.npy', mean_pred)
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/train_label.npy', mean_label)

    print(mean_label)
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())


# def ModelTest(weights_list=None, data_type=None):
#     if data_type is None:
#         data_type = ['alltrain', 'test']
#     from Metric.classification_statistics import get_auc, draw_roc
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     input_shape = (192, 192)
#     batch_size = 2
#     model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
#     bc = BinaryClassification()
#
#     fpr_list, tpr_list, auc_list = [], [], []
#     train_result, test_result = {}, {}
#     for type in data_type:
#         spliter = DataSpliter()
#         # sub_list = spliter.LoadName(data_root / '{}-name.csv'.format(data_type), contain_label=True)
#         sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(type))
#
#         if type == 'test':
#             data = DataManager(sub_list=sub_list)
#             data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
#             data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
#             data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
#             data.AddOne(Image2D(data_root + '/DistanceMap/Test', shape=input_shape, is_roi=True))
#             data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
#             # data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
#             data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
#         else:
#             data = DataManager(sub_list=sub_list)
#             data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
#             data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
#             data.AddOne(Image2D(data_root + '/DwiSlice/', shape=input_shape))
#             data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
#             data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
#             # data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
#             data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
#
#         cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
#         cv_pred_list, cv_label_list = [], []
#
#         for cv_index, cv_folder in enumerate(cv_folder_list):
#             model = ResNeXt(3, 2).to(device)
#             if weights_list is None:
#                 one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
#                                          one.is_file()]
#                 one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
#                 weights_path = one_fold_weights_list[-1]
#             else:
#                 weights_path = weights_list[cv_index]
#
#             print(weights_path.name)
#             model.load_state_dict(torch.load(str(weights_path)))
#
#             pred_list, label_list = [], []
#             model.eval()
#             for inputs, outputs in data_loader:
#                 inputs = MoveTensorsToDevice(inputs, device)
#                 outputs = MoveTensorsToDevice(outputs, device)
#
#                 preds = model(*inputs)[:, 1]
#                 pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
#                 label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
#                 # pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
#                 # label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())
#
#             cv_pred_list.append(pred_list)
#             cv_label_list.append(label_list)
#
#             del model, weights_path
#
#         cv_pred = np.array(cv_pred_list)
#         cv_label = np.array(cv_label_list)
#         mean_pred = np.mean(cv_pred, axis=0)
#         mean_label = np.mean(cv_label, axis=0)
#
#         fpr, tpr, auc = get_auc(mean_pred, mean_label)
#         fpr_list.append(fpr)
#         tpr_list.append(tpr)
#         auc_list.append(auc)
#
#         if type == 'alltrain':
#             train_result = bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist(), threshould=None)
#             print(train_result)
#         elif type == 'test':
#             test_result = bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist(), threshould=train_result['Youden Index'])
#             print(test_result)
#
#     draw_roc(fpr_list, tpr_list, auc_list, name_list=['alltrian', 'test'])
#
#
# def ModelSUH(weights_list=None):
#     from Metric.classification_statistics import get_auc
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     input_shape = (192, 192)
#     batch_size = 2
#     model_folder = model_root + '/ResNeXt_CBAM_CV_20200820'
#     bc = BinaryClassification()
#
#     fpr_list, tpr_list, auc_list = [], [], []
#
#     data = DataManager()
#     data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
#     data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
#     data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
#     # data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
#     # data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Negative'), is_input=False)
#     data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Positive'), is_input=False)
#     data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
#
#     cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
#     cv_pred_list, cv_label_list = [], []
#     cv_fcn_out_list = []
#     for cv_index, cv_folder in enumerate(cv_folder_list):
#         model = ResNeXt(3, 2).to(device)
#         if weights_list is None:
#             one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
#                                      one.is_file()]
#             one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
#             weights_path = one_fold_weights_list[-1]
#         else:
#             weights_path = weights_list[cv_index]
#
#         print(weights_path.name)
#         model.load_state_dict(torch.load(str(weights_path)))
#
#         pred_list, label_list = [], []
#         fcn_out_list = []
#         model.eval()
#         for inputs, outputs in data_loader:
#             inputs = MoveTensorsToDevice(inputs, device)
#             outputs = MoveTensorsToDevice(outputs, device)
#
#             # preds = model(*inputs)[:, 1]
#             model_pred = model(*inputs)
#             preds = model_pred[0][:, 1]
#
#             if isinstance((1 - preds).cpu().data.numpy().squeeze().tolist(), float):
#                     # pred_list.append((1 - preds).cpu().data.numpy().squeeze().tolist())
#                     # label_list.append((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
#                     pred_list.append((preds).cpu().data.numpy().squeeze().tolist())
#                     label_list.append((outputs).cpu().data.numpy().astype(int).squeeze().tolist())
#                     fcn_out_list.append(model_pred[1].cpu().data.numpy().squeeze().tolist())
#             else:
#                     # pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
#                     # label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
#                     pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
#                     label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())
#                     fcn_out_list.extend(model_pred[1].cpu().data.numpy().squeeze().tolist())
#
#         cv_pred_list.append(pred_list)
#         cv_label_list.append(label_list)
#         cv_fcn_out_list.append(fcn_out_list)
#
#         del model, weights_path
#
#     cv_pred = np.array(cv_pred_list)
#     cv_label = np.array(cv_label_list)
#     cv_fcn = np.array(cv_fcn_out_list)
#     mean_pred = np.mean(cv_pred, axis=0)
#     mean_label = np.mean(cv_label, axis=0)
#     mean_fcn_out = np.mean(cv_fcn, axis=0)
#     np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_suh.npy', mean_fcn_out)
#
#     fpr, tpr, auc = get_auc(mean_pred, mean_label)
#     fpr_list.append(fpr)
#     tpr_list.append(tpr)
#     auc_list.append(auc)
#
#     result1 = bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())
#     print(result1)


def ROCofModels(weights_list=None, data_type=['alltrain', 'test']):
    from Metric.classification_statistics import get_auc, draw_roc
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
    bc = BinaryClassification()
    fpr_list, tpr_list, auc_list = [], [], []
    for type in data_type:
        spliter = DataSpliter()
        # sub_list = spliter.LoadName(data_root / '{}-name.csv'.format(data_type), contain_label=True)
        sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(type))

        if type == 'test':
            data = DataManager(sub_list=sub_list)
            data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
            data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
            data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
            data.AddOne(Image2D(data_root + '/DistanceMap/Test', shape=input_shape, is_roi=True))
            data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
            data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        else:
            data = DataManager(sub_list=sub_list)
            data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
            data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
            data.AddOne(Image2D(data_root + '/DwiSlice/', shape=input_shape))
            data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
            data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
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
                pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
                label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
                # pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
                # label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())

            fpr, tpr, auc = get_auc(pred_list, label_list)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            auc_list.append(auc)

            cv_pred_list.append(pred_list)
            cv_label_list.append(label_list)

            del model, weights_path

        cv_pred = np.array(cv_pred_list)
        cv_label = np.array(cv_label_list)
        mean_pred = np.mean(cv_pred, axis=0)
        mean_label = np.mean(cv_label, axis=0)

        fpr, tpr, auc = get_auc(mean_pred, mean_label)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)

    # draw_roc(fpr_list, tpr_list, auc_list, name_list=['cv0', 'cv1', 'cv2', 'cv3', 'cv4', 'alltrian'])
    name_list = ['model1', 'model2', 'model3', 'model4', 'model5', 'model combined']
    for idx in range(len(fpr_list)):
        label = name_list[idx] + ': ' + '%.3f'%auc_list[idx]
        plt.plot(fpr_list[idx], tpr_list[idx], label=label)

    plt.plot([0, 1], [0, 1], '--', color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()


def EnsembleInferencebyCaseName(data_type, weights_list=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (1, 280, 280)
    model_folder = model_root + '/PAGNet0430'
    bc = BinaryClassification()

    ct_folder = os.path.join(data_root, 'ct_slice')
    atten_folder = os.path.join(data_root, 'atten_slice')
    label_path = os.path.join(data_root, '{}_label.csv'.format(data_type))
    info = pd.read_csv(label_path, index_col=0)
    case_name_list = info.index.tolist()

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXtOneInput(1, 2).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        if cv_index == 0:
            case_list = []
        model.eval()
        for index, case_name in enumerate(case_name_list):
            ct_case = os.path.join(ct_folder, '{}.npy'.format(case_name))
            atten_case = os.path.join(atten_folder, '{}.npy'.format(case_name))
            label = torch.tensor(info.loc[case_name]['label'])

            ct, _ = ExtractBlock(np.load(ct_case), patch_size=input_shape, center_point=[-1, -1, -1])
            atten, _ = ExtractBlock(np.load(atten_case), patch_size=input_shape, center_point=[-1, -1, -1])
            ct = ct.astype(np.float32)
            atten = atten.astype(np.float32)

            inputs = [torch.from_numpy(ct[np.newaxis, ...]), torch.from_numpy(atten[np.newaxis, ...])]

            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(label, device)

            model_pred = model(*inputs)
            preds = model_pred[:, 1]
            # pred_list.extend((1 - preds).cpu().data.numpy().squeeze().tolist())
            # label_list.extend((1 - outputs).cpu().data.numpy().astype(int).squeeze().tolist())
            # fcn_out_list.extend(model_pred[1].cpu().data.numpy().squeeze().tolist())

            pred_list.append((preds).cpu().data.numpy())
            label_list.append((outputs).cpu().data.numpy().astype(int))
            if cv_index == 0:
                case_list.append(case_name)

        # bc.Run(pred_list, label_list)

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)
        # cv_fcn_out_list.append(fcn_out_list)

        del model, weights_path

    cv_pred = np.squeeze(np.array(cv_pred_list))
    cv_label = np.squeeze(np.array(cv_label_list))
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)

    if not os.path.exists(r'/home/zhangyihong/Documents/Kindey901/Model/Result'):
        os.mkdir(r'/home/zhangyihong/Documents/Kindey901/Model/Result')

    if data_type == 'test':
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/test.npy', mean_pred)
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/test_label.npy', mean_label)
        df = pd.DataFrame({'CaseName': case_list, 'label': mean_label, 'pred': mean_pred})
        df.to_csv(r'/home/zhangyihong/Documents/Kindey901/Model/Result/test_result.csv', index=False)
    else:
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/train.npy', mean_pred)
        np.save(r'/home/zhangyihong/Documents/Kindey901/Model/Result/train_label.npy', mean_label)
        df = pd.DataFrame({'CaseName': case_list, 'label': mean_label, 'pred': mean_pred})
        df.to_csv(r'/home/zhangyihong/Documents/Kindey901/Model/Result/train_result.csv', index=False)

    print(mean_label)
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())


if __name__ == '__main__':
    # model_root = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred'
    # data_root = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500'
    model_root = r'/home/zhangyihong/Documents/Kindey901/Model'
    data_root = r'/home/zhangyihong/Documents/Kindey901/Kindey_npy'
    # EnsembleInference('test')
    # EnsembleInference('alltrain')
    EnsembleInferencebyCaseName('test')
    EnsembleInferencebyCaseName('alltrain')

    # ROCofModels(data_type=['alltrain'])








