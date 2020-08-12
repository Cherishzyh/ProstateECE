from torch.utils.data import DataLoader

from T4T.Utility.Data import *
from MeDIT.DataAugmentor import random_2d_augment

from Model.ResNetcbam import ResNet, Bottleneck

from NPYFilePath import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

test_dataset = DataManager(random_2d_augment)
test_dataset.AddOne(Image2D(t2_folder, shape=(184, 184)))
test_dataset.AddOne(Image2D(dwi_folder, shape=(184, 184)))
test_dataset.AddOne(Image2D(adc_folder, shape=(184, 184)))
test_dataset.AddOne(Image2D(roi_folder, shape=(184, 184)))
test_dataset.AddOne(Feature(csv_folder))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# with torch.no_grad():
for i, (inputs, outputs) in enumerate(test_loader):
    t2, dwi, adc, roi, ece = inputs[0], inputs[1], inputs[2], inputs[3], np.squeeze(inputs[3], axis=1)
    # np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/t2.npy',
    #         t2)
    # np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/dwi.npy',
    #         dwi)
    # np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/adc.npy',
    #         adc)
    # np.save(r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/roi.npy',
    #         roi)

    inputs = torch.cat([t2, dwi, adc], axis=1)
    inputs = inputs.type(torch.FloatTensor).to(device)
    class_out = model(inputs)