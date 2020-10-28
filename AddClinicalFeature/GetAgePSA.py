import pandas as pd
import os
import numpy as np
from SSHProject.BasicTool.MeDIT.Normalize import NormalizeZ

# JSPH_clinical_report_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ECE-JSPH-clinical_report.csv'
JSPH_clinical_report_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ECE-ROI.csv'
case_csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ece.csv'

clinical_df = pd.read_csv(JSPH_clinical_report_path, encoding='gbk',
                          usecols=['case', 'age', 'psa'], index_col=['case'])
case_name_df = pd.read_csv(case_csv_path, usecols=['case', 'ece'], index_col=['case'])

case_list, age_list, psa_list = [], [], []
for case_name in case_name_df.index:
    case = case_name[:case_name.index('_slice')]
    info = clinical_df.loc[case]
    if isinstance(info['psa'], str):
        if '＞' in info['psa']:
            info['psa'] = info['psa'][1:]
    else:
        print('{} have no psa'.format(case))
    case_list.append(case_name)
    age_list.append(int(info['age']))
    psa_list.append(float(info['psa']))

age_norm = NormalizeZ(np.array(age_list))
psa_norm = NormalizeZ(np.array(psa_list))

age_psa_df = pd.DataFrame(list(map(list, zip(age_norm, psa_norm))), columns=['age', 'psa'], index=[case_list])
age_psa_df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/Age&Psa_norm.csv')
#