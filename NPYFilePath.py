import os

AdcSlice = r'X:\CNNFormatData\ProstateCancerECE\NPY\AdcSlice'
T2Slice = r'X:\CNNFormatData\ProstateCancerECE\NPY\T2Slice'
RoiSlice = r'X:\CNNFormatData\ProstateCancerECE\NPY\RoiSlice'
ProstateSlice = r'X:\CNNFormatData\ProstateCancerECE\NPY\ProstateSlice'
DwiSlice = r'X:\CNNFormatData\ProstateCancerECE\NPY\DwiSlice'
csv_folder = r'X:\CNNFormatData\ProstateCancerECE\NPY\csv\ece.csv'

train_t2_folder = os.path.join(T2Slice, 'Train')
train_dwi_folder = os.path.join(DwiSlice, 'Train')
train_adc_folder = os.path.join(AdcSlice, 'Train')
train_roi_folder = os.path.join(RoiSlice, 'Train')
train_prostate_folder = os.path.join(ProstateSlice, 'Train')

validation_t2_folder = os.path.join(T2Slice, 'Validation')
validation_dwi_folder = os.path.join(DwiSlice, 'Validation')
validation_adc_folder = os.path.join(AdcSlice, 'Validation')
validation_roi_folder = os.path.join(RoiSlice, 'Validation')
validation_prostate_folder = os.path.join(ProstateSlice, 'Validation')

test_t2_folder = os.path.join(T2Slice, 'Test')
test_dwi_folder = os.path.join(DwiSlice, 'Test')
test_adc_folder = os.path.join(AdcSlice, 'Test')
test_roi_folder = os.path.join(RoiSlice, 'Test')
test_prostate_folder = os.path.join(ProstateSlice, 'Test')

model_path = r'Z:\ECE\checkpoint.pt'

