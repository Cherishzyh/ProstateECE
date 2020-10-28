import os
import shutil
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from SSHProject.BasicTool.MeDIT.Augment import *
from SSHProject.BasicTool.MeDIT.Others import MakeFolder, CopyFile, IterateCase

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.Metric import Auc
from SSHProject.CnnTools.T4T.Utility.Loss import BCEFocalLoss
from SSHProject.CnnTools.T4T.Utility.CallBacks import EarlyStopping
from SSHProject.CnnTools.T4T.Utility.Initial import HeWeightInit

# from MyModel.ResnetCBAMDisMapv2 import ResNeXt
from Metric.MyMetric import BinaryClassification
from model import ResNeXt


data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DistanceMap0.2', shape=input_shape))

    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True))
    data.AddOne(Image2D(data_root + '/ProstateSlice', shape=input_shape, is_roi=True))

    data.AddOne(Feature(data_root + '/ece.csv',), is_input=False)

    data.AddOne(Feature(data_root + '/label.csv'), is_input=False)
    data.Balance(Label(data_root + '/label.csv'))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def EnsembleTrain():
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 24
    model_folder = MakeFolder(model_root + '/ResNetcbamOneHot_ensemble')
    CopyFile('ResnetCBAMDisMapv2.py', model_folder / 'ResnetCBAMDisMapv2.py')

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

    spliter = DataSpliter()
    cv_generator = spliter.SplitLabelCV(data_root + '/ece.csv')
    for cv_index, (sub_train, sub_val) in enumerate(cv_generator):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(sub_val, param_config, input_shape, batch_size, True)

        model = ResNeXt(3, 2).to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                               verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss, val_loss = 0., 0.

            model.train()
            pred_list, label_list = [], []
            for ind, (inputs, outputs) in enumerate(train_loader):
                optimizer.zero_grad()

                t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3].to(device)

                input_data = torch.cat([t2, dwi, adc], dim=1)

                input_data = MoveTensorsToDevice(input_data, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(input_data, dismap)

                loss = criterion(preds.float(), outputs[0].long().view(-1))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                pred_list.extend(preds.softmax(dim=1)[:, 1].cpu().data.numpy().tolist())
                label_list.extend(outputs[0].cpu().data.numpy().tolist())

            train_auc = roc_auc_score(label_list, pred_list)

            model.eval()
            pred_list, label_list = [], []
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3].to(device)

                    input_data = torch.cat([t2, dwi, adc], dim=1)

                    input_data = MoveTensorsToDevice(input_data, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    preds = model(input_data, dismap)

                    loss = criterion(preds.float(), outputs[0].long().view(-1))
                    val_loss += loss.item()

                    pred_list.extend(preds.softmax(dim=1)[:, 1].cpu().data.numpy().tolist())
                    label_list.extend(outputs[0].cpu().data.numpy().tolist())

                val_auc = roc_auc_score(label_list, pred_list)

            # Save Tensor Board
            for index, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

            writer.add_scalars('Loss',
                               {'train_loss': train_loss / train_batches,
                                'val_loss': val_loss / val_batches}, epoch + 1)
            writer.add_scalars('Auc',
                               {'train_auc': train_auc,
                                'val_auc': val_auc}, epoch + 1)

            print('Epoch {}: loss: {:.3f}, val-loss: {:.3f}, auc: {:.3f}, val-auc: {:.3f}'.format(
                epoch + 1, train_loss / train_batches, val_loss / val_batches,
                train_auc, val_auc
            ))
            # print('Epoch {}: loss: {:.3f}, val-loss: {:.3f}'.format(
            #     epoch + 1, train_loss / train_batches, val_loss / val_batches))

            scheduler.step(val_loss)
            early_stopping(val_loss, model, (epoch + 1, val_loss))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.flush()
        writer.close()

        del writer, optimizer, scheduler, early_stopping, model


def EnsembleInference(weights_list=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 16
    # model_folder = model_root + '/ResNetcbamOneHot_ensemble'
    model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
    bc = BinaryClassification()

    # spliter = DataSpliter()
    # sub_list = spliter.LoadName(data_root + '/test-name.csv'.format(data_type), contain_label=True)

    # data = DataManager(sub_list=sub_list)
    data = DataManager()
    data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DistanceMap0.2/Test', shape=input_shape, is_roi=True))

    data.AddOne(Feature(data_root + '/ece.csv'), is_input=False)
    # data.AddOne(Label(data_root + '/label.csv'), is_input=False)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
    cv_pred_list, cv_label_list = [], []
    for cv_index, cv_folder in enumerate(cv_folder_list):
        model = ResNeXt(3, num_classes=2).to(device)
        if weights_list is None:
            one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if one.is_file()]
            one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
            weights_path = one_fold_weights_list[-1]
        else:
            weights_path = weights_list[cv_index]

        print(weights_path.name)
        model.load_state_dict(torch.load(str(weights_path)))

        pred_list, label_list = [], []
        model.eval()
        for inputs, outputs in data_loader:
            # t2, dwi, adc, dismap = inputs[0], inputs[1], inputs[2], inputs[3].to(device)
            # input_data = torch.cat([t2, dwi, adc], dim=1).to(device)
            # inputs = [one.to(device) for one in inputs]

            t2, dwi, adc, dismap = inputs[0].to(device), inputs[1].to(device), inputs[2].to(device), inputs[3].to(device)

            outputs = outputs.to(device)

            preds = model(t2, adc, dwi, dismap)

            # print(preds)

            pred_list.extend(preds[:, 0].cpu().data.numpy().squeeze().tolist())
            label_list.extend(outputs.cpu().data.numpy().astype(int).squeeze().tolist())

        # print(pred_list)
        bc.Run(pred_list, label_list)

        cv_pred_list.append(pred_list)
        cv_label_list.append(label_list)

        del model, weights_path

    cv_pred = np.array(cv_pred_list)
    cv_label = np.array(cv_label_list)
    mean_pred = np.mean(cv_pred, axis=0)
    mean_label = np.mean(cv_label, axis=0)

    print(mean_label)
    bc.Run(mean_pred.tolist(), mean_label.astype(int).tolist())


if __name__ == '__main__':
    # EnsembleTrain()
    EnsembleInference()
