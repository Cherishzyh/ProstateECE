import math
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

from SSHProject.BasicTool.MeDIT.Augment import *
from SSHProject.BasicTool.MeDIT.Others import MakeFolder, CopyFile

from SSHProject.CnnTools.T4T.Block.ConvBlock import ConvBn2D
from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.CnnTools.T4T.Utility.CallBacks import EarlyStopping
from SSHProject.CnnTools.T4T.Utility.Initial import HeWeightInit


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 0.5

    def __init__(self, inplanes, planes, stride=1, downsample=None, downstride=2, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.stride = stride

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        if downsample is True:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, downstride),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Layer1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Layer1, self).__init__()
        self.layer1_0 = Bottleneck(inplanes=inplanes, planes=outplanes, downsample=True, downstride=1)
        self.layer1_1 = Bottleneck(inplanes=outplanes, planes=outplanes)
        self.layer1_2 = Bottleneck(inplanes=outplanes, planes=outplanes)

    def forward(self, x):
        x = self.layer1_0(x)
        x = self.layer1_1(x)
        x = self.layer1_2(x)
        return x


class Layer2(nn.Module):
    def __init__(self, inplanes, outplanes):
        self.inplanes = inplanes

        super(Layer2, self).__init__()
        self.layer2_0 = Bottleneck(inplanes=inplanes, planes=outplanes, downsample=True, stride=2)
        self.layer2_1 = Bottleneck(inplanes=outplanes, planes=outplanes)
        self.layer2_2 = Bottleneck(inplanes=outplanes, planes=outplanes)
        self.layer2_3 = Bottleneck(inplanes=outplanes, planes=outplanes)

    def forward(self, x):
        x = self.layer2_0(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer2_3(x)

        return x


class Layer3(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Layer3, self).__init__()
        self.layer3_0 = Bottleneck(inplanes=inplanes, planes=outplanes, downsample=True, stride=2)
        self.layer3_1 = Bottleneck(inplanes=outplanes, planes=outplanes)
        self.layer3_2 = Bottleneck(inplanes=outplanes, planes=outplanes)
        self.layer3_3 = Bottleneck(inplanes=outplanes, planes=outplanes)

    def forward(self, x):
        x = self.layer3_0(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer3_3(x)

        return x


class Layer4(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Layer4, self).__init__()
        self.layer4_0 = Bottleneck(inplanes=inplanes, planes=outplanes, downsample=True, stride=2)
        self.layer4_1 = Bottleneck(inplanes=outplanes, planes=outplanes)
        self.layer4_2 = Bottleneck(inplanes=outplanes, planes=outplanes)

    def forward(self, x):
        x = self.layer4_0(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)

        return x


class ResnetCBAM(nn.Module):
    def __init__(self, in_channels, num_classes, inplanes=32):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(ResnetCBAM, self).__init__()

        self.conv1 = ConvBn2D(in_channels, inplanes)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = Layer1(inplanes, inplanes * 2)
        self.layer2 = Layer2(inplanes * 2, inplanes * 4)
        self.layer3 = Layer3(inplanes * 4, inplanes * 6)
        self.layer4 = Layer4(inplanes * 6, inplanes * 8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(inplanes * 8, inplanes),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(inplanes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, t2, adc, dwi):
        inputs = torch.cat([t2, adc, dwi], dim=1)
        x = self.conv1(inputs)
        x = self.maxpool1(x)  # shape = (92, 92)

        x = self.layer1(x)  # shape = (92, 92)
        x = self.layer2(x)  # shape = (46, 46)
        x = self.layer3(x)  # shape = (23, 23)
        x = self.layer4(x)  # shape = (12, 12)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_fc1 = self.fc1(x)
        x = self.fc2(x_fc1)
        return torch.softmax(x, dim=1)


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ResnetCBAM(3, num_classes=2).to(device)
    print(model)
    inputs = torch.randn(1, 1, 184, 184).to(device)
    prediction = model(inputs, inputs, inputs)
    print(prediction.shape)


model_root = r'/home/zhangyihong/Documents/ProstateECE/BaseModel'
data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred'


# def ClearGraphPath(graph_path):
#     if not os.path.exists(graph_path):
#         os.mkdir(graph_path)
#     else:
#         shutil.rmtree(graph_path)
#         os.mkdir(graph_path)


def _GetLoader(sub_list, aug_param_config, input_shape, batch_size, shuffle):
    data = DataManager(sub_list=sub_list, augment_param=aug_param_config)

    data.AddOne(Image2D(data_root + '/Train/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/Train/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/Train/DwiSlice', shape=input_shape))
    data.AddOne(Label(data_root + '/ece.csv'), is_input=False)
    data.Balance(Label(data_root + '/ece.csv'))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    batches = np.ceil(len(data.indexes) / batch_size)
    return loader, batches


def EnsembleTrain():
    torch.autograd.set_detect_anomaly(True)

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    total_epoch = 10000
    batch_size = 24
    model_folder = MakeFolder(model_root + '/ResnetCBAM')

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

    sub_train_path = data_root + '/train_name_basemodel.csv'
    sub_val_path = data_root + '/val_name_basemodel.csv'
    sub_train = pd.read_csv(sub_train_path).values.tolist()[0]
    sub_val = pd.read_csv(sub_val_path).values.tolist()[0]
    train_loader, train_batches = _GetLoader(sub_train, param_config, input_shape, batch_size, True)
    val_loader, val_batches = _GetLoader(sub_val, param_config, input_shape, batch_size, True)

    model = ResnetCBAM(3, 2).to(device)
    model.apply(HeWeightInit)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    cr = torch.nn.NLLLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5,
                                                           verbose=True)
    early_stopping = EarlyStopping(store_path=os.path.join(model_folder, '{}-{:.6f}.pt'), patience=50, verbose=True)
    writer = SummaryWriter(log_dir=os.path.join(model_folder, 'log'), comment='Net')

    for epoch in range(total_epoch):
        train_loss, val_loss = 0., 0.

        model.train()
        pred_list, label_list = [], []
        for ind, (inputs, outputs) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = MoveTensorsToDevice(inputs, device)
            outputs = MoveTensorsToDevice(outputs, device)

            preds = model(*inputs)

            loss = cr(preds, outputs.long())

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_list.extend(preds[:, 1].cpu().data.numpy().tolist())
            label_list.extend(outputs.cpu().data.numpy().tolist())

        train_auc = roc_auc_score(label_list, pred_list)

        model.eval()
        pred_list, label_list = [], []
        with torch.no_grad():
            for ind, (inputs, outputs) in enumerate(val_loader):
                inputs = MoveTensorsToDevice(inputs, device)
                outputs = MoveTensorsToDevice(outputs, device)

                preds = model(*inputs)

                loss = cr(preds, outputs.long())

                val_loss += loss.item()

                pred_list.extend(preds[:, 1].cpu().data.numpy().tolist())
                label_list.extend(outputs.cpu().data.numpy().tolist())

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

        scheduler.step(val_loss)
        early_stopping(val_loss, model, (epoch + 1, val_loss))

        if early_stopping.early_stop:
            print("Early stopping")
            break

        writer.flush()
        writer.close()

if __name__ == '__main__':
    # test()
    EnsembleTrain()