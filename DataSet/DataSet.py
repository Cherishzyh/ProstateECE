from torch.utils.data import DataLoader

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.BasicTool.MeDIT.Augment import config_example

# 数据拆分
spliter = DataSpliter()
spliter.SplitLabel(r'w:\CNNFormatData\ProsateX\cs_label.csv', ratio=0.8, store_name=('train', 'test'))


sub_list = spliter.LoadName(file_path)

train_dataset = DataManager(sub_list=sub_list, augment_param=config_example)

###########################################################
# input image
train_dataset.AddOne(Image2D(train_t2_folder, shape=shape))
train_dataset.AddOne(Image2D(train_dwi_folder, shape=shape))
train_dataset.AddOne(Image2D(train_adc_folder, shape=shape))
# input roi
train_dataset.AddOne(Image2D(train_roi_folder, shape=shape, is_roi=True))
train_dataset.AddOne(Image2D(train_prostate_folder, shape=shape, is_roi=True))

# label, not input
train_dataset.AddOne(Label(label_folder, label_tag='****'), is_input=False)
# or
# train_dataset.AddOne(Feature(ece_folder), is_input=False)

# data balance
train_dataset.Balance(Label(label_folder, label_tag='****'))

###########################################################
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)


for i, (inputs, outputs) in enumerate(train_loader):
    t2, dwi, adc, pca, prostate = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
    ece = outputs
