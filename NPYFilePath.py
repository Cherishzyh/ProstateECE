import os


# data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYOnehot'
# data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY'
data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain'
model_original_folder = r'/home/zhangyihong/Documents/ProstateECE/Model'

model_folder = os.path.join(model_original_folder, 'MultiTaskPreTrain')

t2_folder = os.path.join(data_folder, 'T2Slice')
dwi_folder = os.path.join(data_folder, 'DwiSlice')
adc_folder = os.path.join(data_folder, 'AdcSlice')
roi_folder = os.path.join(data_folder, 'RoiSlice')
prostate_folder = os.path.join(data_folder, 'ProstateSlice')
ece_folder = os.path.join(data_folder, 'csv')

train_t2_folder = os.path.join(t2_folder, 'Train')
train_dwi_folder = os.path.join(dwi_folder, 'Train')
train_adc_folder = os.path.join(adc_folder, 'Train')
train_roi_folder = os.path.join(roi_folder, 'Train')
train_prostate_folder = os.path.join(prostate_folder, 'Train')

pre_train_t2_folder = os.path.join(t2_folder, 'PreTrain')
pre_train_dwi_folder = os.path.join(dwi_folder, 'PreTrain')
pre_train_adc_folder = os.path.join(adc_folder, 'PreTrain')
pre_train_roi_folder = os.path.join(roi_folder, 'PreTrain')
pre_train_prostate_folder = os.path.join(prostate_folder, 'PreTrain')

validation_t2_folder = os.path.join(t2_folder, 'Validation')
validation_dwi_folder = os.path.join(dwi_folder, 'Validation')
validation_adc_folder = os.path.join(adc_folder, 'Validation')
validation_roi_folder = os.path.join(roi_folder, 'Validation')
validation_prostate_folder = os.path.join(prostate_folder, 'Validation')

pre_validation_t2_folder = os.path.join(t2_folder, 'PreValid')
pre_validation_dwi_folder = os.path.join(dwi_folder, 'PreValid')
pre_validation_adc_folder = os.path.join(adc_folder, 'PreValid')
pre_validation_roi_folder = os.path.join(roi_folder, 'PreValid')
pre_validation_prostate_folder = os.path.join(prostate_folder, 'PreValid')

test_t2_folder = os.path.join(t2_folder, 'Test')
test_dwi_folder = os.path.join(dwi_folder, 'Test')
test_adc_folder = os.path.join(adc_folder, 'Test')
test_roi_folder = os.path.join(roi_folder, 'Test')
test_prostate_folder = os.path.join(prostate_folder, 'Test')

csv_folder = os.path.join(ece_folder, 'ece.csv')

model_path = os.path.join(model_folder, 'checkpoint.pt')
graph_path = os.path.join(model_folder, 'logs')


# t2_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/t2'
# dwi_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/dwi'
# adc_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/adc'
# roi_folder = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/roi'
# csv = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/csv/ece.csv'