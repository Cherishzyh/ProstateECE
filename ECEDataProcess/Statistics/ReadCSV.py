import os
import pandas as pd


def ReadCSV():
    folder_path = r'X:\PrcoessedData\ProstateCancerECE'
    store_path = r'C:\Users\ZhangYihong\Desktop\DataInformation.csv'
    case_list = sorted(os.listdir(folder_path))

    for case in case_list:
        case_path = os.path.join(folder_path, case)
        roi_path = os.path.join(case_path, 'roi.csv')
        if not os.path.exists(roi_path):
            print('roi of {} is not exists'.format(case))
            print('1')
            continue
        one_label_df = pd.read_csv(roi_path)
        one_label_df.insert(0, 'case', case)
        if case == case_list[0]:
            one_label_df.to_csv(store_path, mode='a+', index=False, header=True)
        else:
            one_label_df.to_csv(store_path, mode='a+', index=False, header=False)


if __name__ == '__main__':
    # folder_path = r'X:\PrcoessedData\ProstateCancerECE\BAO ZHENG LI'
    # store_path = r'C:\Users\ZhangYihong\Desktop\BAO ZHENG LI.csv'
    #
    # roi_path = os.path.join(folder_path, 'roi.csv')
    # one_label_df = pd.read_csv(roi_path)
    # one_label_df.insert(0, 'case', 'BAO ZHENG LI')
    #
    # one_label_df.to_csv(store_path, mode='a+', index=False, header=True)

    # one_label_df.to_csv(store_path, mode='a+', index=case, header=False)
    ReadCSV()