import os
import pandas as pd
import pydicom

from MeDIT.Log import CustomerCheck, Eclog


def GetPath(case_folder):
    for root, dirs, files in os.walk(case_folder):
        if len(files) > 3:
            return root, dirs, files
        elif dirs == files == []:
            return root, dirs, files

def ReadDICOMHead(data_path):
    # name, age, male,
    try:
        ds = pydicom.read_file(data_path)
        return ds.PatientID, ds.PatientSex, ds.PatientAge
    except Exception as e:
        print(data_path)

if __name__ == '__main__':
    raw_folder = r'X:\RawData\ProstateCancerECE\PCa-RP'
    store_folder = r'X:\PrcoessedData\ProstateCancerECE'
    failed_folder = r'X:\PrcoessedData\ProstateCancerECE'

    log = CustomerCheck(os.path.join(failed_folder, 'dicom_head_failed_log.csv'), patient=1, data={'State': [], 'Info': []})
    # eclog = Eclog(os.path.join(failed_folder, 'failed_log_details.log')).GetLogger()

    for case in sorted(os.listdir(raw_folder)):
        des_case_folder = os.path.join(store_folder, case)
        csv_store_path = os.path.join(des_case_folder, 'PatientInfo.csv')

        case_folder = os.path.join(raw_folder, case)
        case_path, dirs, files = GetPath(case_folder)
        print(case_path, dirs, files)
        if dirs == []:
            log.AddOne(case, {'State': 'Have no dicom data'})
            print('{} is failed to find dicom data.'.format(case))
            continue
        dicom_folder = os.path.join(case_path, dirs[0])
        dicom_file = os.listdir(dicom_folder)
        dicom_path = os.path.join(dicom_folder, dicom_file[0])

        try:
            ds = pydicom.read_file(dicom_path)
            PatientID = ds.PatientID
            PatientSex = ds.PatientSex
            PatientAge = ds.PatientAge
        except Exception as e:
            continue

        info_dict = {'PatientID': PatientID, 'PatientSex': PatientSex, 'PatientAge': PatientAge}

        one_label_df = pd.DataFrame(info_dict, index=['Information'])

        # one_label_df.to_csv(csv_store_path)
