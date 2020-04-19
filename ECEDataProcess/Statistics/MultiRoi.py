import csv
import os
import pandas as pd
import matplotlib.pyplot as plt

from MeDIT.SaveAndLoad import LoadNiiData
from MeDIT.Visualization import Imshow3DArray
from MeDIT.Normalize import Normalize01

from FilePath import process_folder

def ShowRoi(process_folder):
    for case in sorted(os.listdir(process_folder)):
        print(case)
        case = 'CDM^chen di ming'
        case_folder = os.path.join(process_folder, case)
        csv_path = os.path.join(case_folder, 'roi.csv')
        t2_path = os.path.join(case_folder, 't2.nii')

        roi_list = ['roi.nii', 'roi0.nii', 'roi1.nii', 'roi2.nii']

        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            column = [row['ECE'] for row in reader]
            print(column)

        _, t2, _ = LoadNiiData(t2_path)
        Imshow3DArray(Normalize01(t2.transpose((1, 2, 0))))
        for roi in roi_list:
            roi_path = os.path.join(case_folder, roi)
            if not os.path.exists(roi_path):
                continue
            _, roi, _ = LoadNiiData(roi_path)
            try:
                Imshow3DArray(Normalize01(t2.transpose((1, 2, 0))), roi=Normalize01(roi.transpose(1, 2, 0)))
            except Exception as e:
                print('failed')

def ShowOne():
    path = r'X:\PrcoessedData\ProstateCancerECE\FCX^fang chun xiang\t2.nii'
    roi0_path = r'X:\PrcoessedData\ProstateCancerECE\FCX^fang chun xiang\roi0.nii'
    roi1_path = r'X:\PrcoessedData\ProstateCancerECE\FCX^fang chun xiang\roi1.nii'
    # roi2_path = r'X:\PrcoessedData\ProstateCancerECE\CDM^chen di ming\roi2.nii'
    _, data, _ = LoadNiiData(path)
    _, roi0, _ = LoadNiiData(roi0_path)
    _, roi1, _ = LoadNiiData(roi1_path)
    # _, roi2, _ = LoadNiiData(roi2_path)

    data = data.transpose((1, 2, 0))
    roi0 = roi0.transpose((1, 2, 0))
    roi1 = roi1.transpose((1, 2, 0))
    # roi2 = roi2.transpose((1, 2, 0))

    Imshow3DArray(Normalize01(data), roi=[roi0, roi1])

def StatisticsMultiRoi():
    csv_store_path = r'C:\Users\ZhangYihong\Desktop\lalalal.csv'
    for case in sorted(os.listdir(process_folder)):

        case_folder = os.path.join(process_folder, case)
        csv_path = os.path.join(case_folder, 'roi.csv')
        roi_path = os.path.join(case_folder, 'roi0.nii')
        if os.path.exists(roi_path):
            print(case)
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                PIRADS = [row['PIRADS'] for row in reader]
                # column = [row['ECE'] for row in reader]
            print(PIRADS)
            if len(PIRADS) == 2:
                one_label_df = pd.DataFrame(PIRADS, index=[case+'-roi0', case+'-roi1'])
                one_label_df.to_csv(csv_store_path, mode='a', header=False)
            if len(PIRADS) == 3:
                one_label_df = pd.DataFrame(PIRADS, index=[case+'-roi0', case+'-roi1', case+'-roi2'])
                one_label_df.to_csv(csv_store_path, mode='a', header=False)


def StatisticsMultiRoi2():
    roi_2_number = 0
    pirads_diff = 0
    ece_diff = 0
    same = 0
    for case in sorted(os.listdir(process_folder)):

        case_folder = os.path.join(process_folder, case)
        csv_path = os.path.join(case_folder, 'roi.csv')
        roi_path = os.path.join(case_folder, 'roi0.nii')

        if os.path.exists(roi_path):
            roi_2_number += 1
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                PIRADS = [row['PIRADS'] for row in reader]
            with open(csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                ECE = [row['ECE'] for row in reader]

            if len(PIRADS) == 2:
                if PIRADS[0] == PIRADS[1]:
                    pirads_diff += 1
                if ECE[0] == ECE[1]:
                    ece_diff += 1
                if PIRADS[0] == PIRADS[1] and ECE[0] == ECE[1]:
                    same += 1
            elif len(PIRADS) == 3:
                if PIRADS[0] == PIRADS[1] == PIRADS[2]:
                    pirads_diff += 1
                if ECE[0] == ECE[1] == ECE[2]:
                    ece_diff += 1
                if PIRADS[0] == PIRADS[1] == PIRADS[2] and ECE[0] == ECE[1] == ECE[2]:
                    same += 1
            elif len(PIRADS) == 4:
                if PIRADS[0] == PIRADS[1] == PIRADS[2] == PIRADS[3]:
                    pirads_diff += 1
                if ECE[0] == ECE[1] == ECE[2] == ECE[3]:
                    ece_diff += 1
                if PIRADS[0] == PIRADS[1] == PIRADS[2] == PIRADS[3] and ECE[0] == ECE[1] == ECE[2] == ECE[3]:
                    same += 1
            else:
                print(case, len(PIRADS))

    print(roi_2_number, pirads_diff, ece_diff, same)


if __name__ == '__main__':
    ShowOne()


