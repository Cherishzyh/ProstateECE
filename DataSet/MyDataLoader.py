from torch.utils.data import DataLoader

from SSHProject.CnnTools.T4T.Utility.Data import *
from SSHProject.BasicTool.MeDIT.Augment import config_example

from NPYFilePath import *


def LoadTVData(folder, shape=(184, 184), is_test=False, setname=None):

    if setname is None:
        setname = ['Train', 'Validation']

    t2_folder = os.path.join(folder, 'T2Slice')
    dwi_folder = os.path.join(folder, 'DwiSlice')
    adc_folder = os.path.join(folder, 'AdcSlice')
    roi_folder = os.path.join(folder, 'RoiSlice')
    prostate_folder = os.path.join(folder, 'ProstateSlice')
    csv_folder = os.path.join(folder, 'csv')
    ece_folder = os.path.join(csv_folder, 'ece.csv')
    # label_folder = os.path.join(csv_folder, 'label.csv')

    train_t2_folder = os.path.join(t2_folder, setname[0])
    train_dwi_folder = os.path.join(dwi_folder, setname[0])
    train_adc_folder = os.path.join(adc_folder, setname[0])
    train_roi_folder = os.path.join(roi_folder, setname[0])
    train_prostate_folder = os.path.join(prostate_folder, setname[0])

    validation_t2_folder = os.path.join(t2_folder, setname[1])
    validation_dwi_folder = os.path.join(dwi_folder, setname[1])
    validation_adc_folder = os.path.join(adc_folder, setname[1])
    validation_roi_folder = os.path.join(roi_folder, setname[1])
    validation_prostate_folder = os.path.join(prostate_folder, setname[1])

    if is_test:
        train_dataset = DataManager()
        validation_dataset = DataManager()
    else:
        train_dataset = DataManager(config_example)
        validation_dataset = DataManager(config_example)

    ###########################################################
    train_dataset.AddOne(Image2D(train_t2_folder, shape=shape))
    train_dataset.AddOne(Image2D(train_dwi_folder, shape=shape))
    train_dataset.AddOne(Image2D(train_adc_folder, shape=shape))

    train_dataset.AddOne(Image2D(train_roi_folder, shape=shape, is_roi=True))
    train_dataset.AddOne(Image2D(train_prostate_folder, shape=shape, is_roi=True))

    train_dataset.AddOne(Feature(ece_folder), is_input=False)

    ###########################################################
    validation_dataset.AddOne(Image2D(validation_t2_folder, shape=shape))
    validation_dataset.AddOne(Image2D(validation_dwi_folder, shape=shape))
    validation_dataset.AddOne(Image2D(validation_adc_folder, shape=shape))

    validation_dataset.AddOne(Image2D(validation_roi_folder, shape=shape, is_roi=True))
    validation_dataset.AddOne(Image2D(validation_prostate_folder, shape=shape, is_roi=True))

    validation_dataset.AddOne(Feature(ece_folder), is_input=False)

    ###########################################################
    if is_test:
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=12, shuffle=True)

    return train_loader, validation_loader


def LoadTestData(folder, shape=(184, 184)):

    t2_folder = os.path.join(folder, 'T2Slice')
    dwi_folder = os.path.join(folder, 'DwiSlice')
    adc_folder = os.path.join(folder, 'AdcSlice')
    roi_folder = os.path.join(folder, 'RoiSlice')
    prostate_folder = os.path.join(folder, 'ProstateSlice')
    csv_folder = os.path.join(folder, 'csv')
    ece_folder = os.path.join(csv_folder, 'ece.csv')

    test_t2_folder = os.path.join(t2_folder, 'Test')
    test_dwi_folder = os.path.join(dwi_folder, 'Test')
    test_adc_folder = os.path.join(adc_folder, 'Test')
    test_roi_folder = os.path.join(roi_folder, 'Test')
    test_prostate_folder = os.path.join(prostate_folder, 'Test')

    test_dataset = DataManager()

    test_dataset.AddOne(Image2D(test_t2_folder, shape=shape))
    test_dataset.AddOne(Image2D(test_dwi_folder, shape=shape))
    test_dataset.AddOne(Image2D(test_adc_folder, shape=shape))

    test_dataset.AddOne(Image2D(test_roi_folder, shape=shape, is_roi=True))
    test_dataset.AddOne(Image2D(test_prostate_folder, shape=shape, is_roi=True))

    test_dataset.AddOne(Feature(ece_folder), is_input=False)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return test_loader