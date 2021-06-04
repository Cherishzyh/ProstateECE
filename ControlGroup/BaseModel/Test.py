import torch
from torch.utils.data import DataLoader

from SSHProject.CnnTools.T4T.Utility.Data import *

from SSHProject.BasicTool.MeDIT.Statistics import BinaryClassification


from ProstateECE.ControlGroup.BaseModel.Resnet import Resnet
from ProstateECE.ControlGroup.BaseModel.Resnext import Resnext
from ProstateECE.ControlGroup.BaseModel.ResnetCBAM import ResnetCBAM
from ProstateECE.ControlGroup.BaseModel.ResnextCBAM import ResnextCBAM

def ModelTest(model_name, data_type, weights_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    model_folder = os.path.join(r'/home/zhangyihong/Documents/ProstateECE/BaseModel', model_name)
    data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYMaxPred'
    bc = BinaryClassification()

    if data_type == 'test':
        data_folder = os.path.join(data_root, "Test")
        data = DataManager()
    else:
        data_folder = os.path.join(data_root, "Train")
        spliter = DataSpliter()
        sub_list = spliter.LoadName(data_root + '/{}_name_basemodel.csv'.format(data_type))
        data = DataManager(sub_list=sub_list)

    data.AddOne(Image2D(data_folder + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_folder + '/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_folder + '/DwiSlice', shape=input_shape))
    data.AddOne(Label(data_root + '/ece.csv'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = ResnextCBAM(3, 2).to(device)

    model.load_state_dict(torch.load(os.path.join(model_folder, weights_name)))

    pred_list, label_list = [], []
    model.eval()
    for inputs, outputs in data_loader:
        inputs = MoveTensorsToDevice(inputs, device)
        outputs = MoveTensorsToDevice(outputs, device)

        preds = model(*inputs)[:, 1]
        pred_list.extend((preds).cpu().data.numpy().squeeze().tolist())
        label_list.extend((outputs).cpu().data.numpy().astype(int).squeeze().tolist())
    #
    #
    # fpr, tpr, auc = get_auc(pred_list, label_list)

    result = bc.Run(pred_list, label_list)
    print(result)


if __name__ == '__main__':
    model_name = ['Resnet', 'ResnetCBAM', 'Resnext', 'ResnextCBAM']
    data_type = ['train', 'val', 'test']
    weights_name = ['27--6.102111.pt', '51--6.044856.pt', '21--5.896627.pt', '82--6.052718.pt']

    ModelTest(model_name[3], data_type[0], weights_name[3])