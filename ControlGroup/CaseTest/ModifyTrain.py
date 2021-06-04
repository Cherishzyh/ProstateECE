import shutil
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from SSHProject.BasicTool.MeDIT.Augment import *
from SSHProject.BasicTool.MeDIT.Others import MakeFolder, CopyFile

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.CallBacks import EarlyStopping
from SSHProject.CnnTools.T4T.Utility.Initial import HeWeightInit

from ControlGroup.CaseTest.ModifyModel import ResNeXt

model_root = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred/Model'
data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred'


def ClearGraphPath(graph_path):
    if not os.path.exists(graph_path):
        os.mkdir(graph_path)
    else:
        shutil.rmtree(graph_path)
        os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)
    data.AddOne(Image2D(data_root + '/Train/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/Train/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/Train/DwiSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/Train/DistanceMap', shape=input_shape, is_roi=True))
    data.AddOne(Feature(data_root + '/clinical.csv'), is_input=True)
    data.AddOne(Label(data_root + '/train_ece.csv'), is_input=False)
    data.Balance(Label(data_root + '/train_ece.csv'))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def EnsembleTrain():
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 24
    model_folder = MakeFolder(model_root + '/PAGNet_MaxPred')
    CopyFile('ModifyModel.py', model_folder / 'model.py')

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
    cv_generator = spliter.SplitLabelCV(data_root + '/train_ece.csv', store_root=model_folder)
    for cv_index, (sub_train, sub_val) in enumerate(cv_generator):
        sub_model_folder = MakeFolder(model_folder / 'CV_{}'.format(cv_index))
        train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
        val_loader, val_batches = _GetLoader(sub_val, param_config, input_shape, batch_size, True)

        model = ResNeXt(3, 2, 7).to(device)
        model.apply(HeWeightInit)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        criterion = torch.nn.NLLLoss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                               verbose=True)
        early_stopping = EarlyStopping(store_path=str(sub_model_folder / '{}-{:.6f}.pt'), patience=50, verbose=True)
        writer = SummaryWriter(log_dir=str(sub_model_folder / 'log'), comment='Net')

        for epoch in range(total_epoch):
            train_loss1, val_loss1 = 0., 0.
            train_loss2, val_loss2 = 0., 0.
            train_loss, val_loss = 0., 0.

            model.train()
            pred1_list, pred2_list, label_list = [], [], []
            for ind, (inputs, outputs) in enumerate(train_loader):
                optimizer.zero_grad()

                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds1, preds2 = model(*inputs)

                loss1 = criterion(preds1, outputs.long())
                loss2 = criterion(preds2, outputs.long())
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_loss1 += loss1.item()
                train_loss2 += loss2.item()

                pred1_list.extend(preds1[:, 1].cpu().data.numpy().tolist())
                pred2_list.extend(preds2[:, 1].cpu().data.numpy().tolist())
                label_list.extend(outputs.cpu().data.numpy().tolist())

            train_auc1 = roc_auc_score(label_list, pred1_list)
            train_auc2 = roc_auc_score(label_list, pred2_list)

            model.eval()
            pred1_list, pred2_list, label_list = [], [], []
            with torch.no_grad():
                for ind, (inputs, outputs) in enumerate(val_loader):
                    inputs = MoveTensorsToDevice(inputs, device)
                    outputs = MoveTensorsToDevice(outputs, device)

                    preds1, preds2 = model(*inputs)

                    loss1 = criterion(preds1, outputs.long())
                    loss2 = criterion(preds2, outputs.long())
                    loss = loss1 + loss2

                    val_loss += loss.item()
                    val_loss1 += loss1.item()
                    val_loss2 += loss2.item()

                    pred1_list.extend(preds1[:, 1].cpu().data.numpy().tolist())
                    pred2_list.extend(preds2[:, 1].cpu().data.numpy().tolist())
                    label_list.extend(outputs.cpu().data.numpy().tolist())

                val_auc1 = roc_auc_score(label_list, pred1_list)
                val_auc2 = roc_auc_score(label_list, pred2_list)

            # Save Tensor Board
            for index, (name, param) in enumerate(model.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name + '_data', param.cpu().data.numpy(), epoch + 1)

            writer.add_scalars('Loss',
                               {'train_loss': train_loss / train_batches,
                                'val_loss': val_loss / val_batches}, epoch + 1)

            writer.add_scalars('Loss',
                               {'train_loss1': train_loss1 / train_batches, 'train_loss2': train_loss2 / train_batches,
                                'val_loss1': val_loss1 / val_batches, 'val_loss2': val_loss2 / val_batches}, epoch + 1)

            writer.add_scalars('Auc',
                               {'train_auc1': train_auc1, 'train_auc2': train_auc2,
                                'val_auc1': val_auc1, 'val_auc2': val_auc2}, epoch + 1)

            print('Epoch {}: loss: {:.3f}, val-loss: {:.3f}, auc1: {:.3f}, val-auc1: {:.3f} auc2: {:.3f}, val-auc2: {:.3f}'.format(
                epoch + 1, train_loss / train_batches, val_loss / val_batches,
                train_auc1, val_auc1, train_auc2, val_auc2
            ))

            scheduler.step(val_loss)
            early_stopping(val_loss, model, (epoch + 1, val_loss))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            writer.flush()
        writer.close()

        del writer, optimizer, scheduler, early_stopping, model


if __name__ == '__main__':
    EnsembleTrain()
