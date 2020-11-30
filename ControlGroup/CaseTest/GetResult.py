import os
import numpy as np
import pandas as pd

from BasicTool.MeDIT.SaveAndLoad import LoadH5
from BasicTool.MeDIT.Statistics import BinaryClassification


def ComputeAUC(data_path):
    case_list = sorted(os.listdir(data_path))
    pred_list, label_list, case_name = [], [], []
    for case in case_list:
        case_path = os.path.join(data_path, case)
        slice_preds, label = LoadH5(case_path, tag=['prediction', 'label'], data_type=[np.float32, np.uint8])
        slice_preds = [1 - pred for pred in slice_preds]

        pred_list.append(np.max(np.array(slice_preds)))
        label_list.append(label.astype(int).tolist())
        case_name.append(case[:case.index('.h5')])

    bc = BinaryClassification()
    bc.Run(pred_list, label_list)
    return pred_list, label_list, case_name

if __name__ == '__main__':
    case_folder = r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\SUH'
    pred_list, label_list, case_name = ComputeAUC(case_folder)
    df = pd.DataFrame({'case': case_name, 'Label': label_list, 'PAGNet': pred_list}, index=None)
    df.to_csv(r'X:\CNNFormatData\ProstateCancerECE\Result\CaseH5\PAGNet\suh.csv')
