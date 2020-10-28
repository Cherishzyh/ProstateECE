import os
import shutil
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt

# feature_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map'
# feature_list = os.listdir(feature_path)
# feature_list.sort()
# data = []
# for case in feature_list:
#     npy_case = os.path.join(feature_path, case)
#     data = (np.load(npy_case))
#     data = np.squeeze(data)
#     plt.imshow(data, cmap='gray')
#     plt.show()
# feature_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/try/feature_map/t2.npy'
# t2_path = r'/home/zhangyihong/Documents/zhangyihong/Documents/ProstateECE/NPY/T2Slice/Test/BHX^bao han xiu ^^6875-5_slice9.npy'
# data = np.load(feature_path)
# data = np.squeeze(data)
# t2_data = np.load(t2_path)
# t2_data = np.squeeze(t2_data)
# plt.subplot(121)
# plt.imshow(data, cmap='gray')
# plt.subplot(122)
# plt.imshow(t2_data, cmap='gray')
# plt.show()

# case_name = 'DSR^dai shou rong_slice16.npy'
# case_name_before = 'DSR^dai shou rong_slice14.npy'
#
# data_path = r'/home/zhangyihong/Documents/ProstateECE/NPYPreTrain'
# data_path_list = os.listdir(data_path)
# for folder in data_path_list:
#     case_folder = os.path.join(os.path.join(data_path, folder), 'PreTrain')
#     case_path_before = os.path.join(case_folder, case_name_before)
#     case_path = os.path.join(case_folder, case_name)
#     if os.path.exists(case_path_before):
#
#         shutil.copy(case_path_before, case_path)
#         print(case_path_before)
#         print(case_path)
#         os.remove(case_path_before)

# data_folder = r'/home/zhangyihong/Documents/ProstateECE/NPY/AdcSlice'
# # train_folder = os.path.join(data_folder, 'Train')
# # valid_folder = os.path.join(data_folder, 'Validation')
# # test_folder = os.path.join(data_folder, 'Test')
#
# train_list = os.listdir(data_folder)
# train_list.sort()
# df = pd.DataFrame(index=['test'])
# for index, case in enumerate(train_list):
#     # df = pd.DataFrame(case, index=['train'], columns=[index])
#     df[index] = [case[:case.index('.npy')]]
# df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/NPY/alltrain-name.csv', mode='a+')
# print('done')


# npy_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\T2Slice'
# label_path = r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\ece.csv'
# # df = pd.DataFrame(index=['test'])
# info = pd.read_csv(label_path, index_col=0)
# for index, case in enumerate(os.listdir(npy_path)):
#     case_name = case[:case.index('.npy')]
#     ece = info.loc[case_name, '0']
#     print(index, case_name)


# orginal_csv_path = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/ECE-JSPH-clinical.csv'
# df = pd.read_csv(orginal_csv_path, index_col='case')
#
# case_list = os.listdir(r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide/AdcSlice/Test')
# # case_name = [case[: case.index('_slice')] for case in case_list]
# new_name = []
# for index in df.index:
#     # info = df.loc[index]
#     for case in case_list:
#         case_name = case[: case.index('_slice')]
#         if case_name == index:
#             new_name.append(case)
# df['new_name'] = new_name
    # df[case] = case_name
    # print()
# import matplotlib.pyplot as plt
# from scipy.stats import wilcoxon
#
# import torch
# from torch.utils.data import DataLoader
#
# from SSHProject.CnnTools.T4T.Utility.Data import *
# from SSHProject.BasicTool.MeDIT.Others import IterateCase
# from SSHProject.BasicTool.MeDIT.ArrayProcess import ExtractBlock
# from SSHProject.BasicTool.MeDIT.Statistics import BinaryClassification
# bc = BinaryClassification()
#
# model_root = r'/home/zhangyihong/Documents/ProstateECE/Model'
# data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# input_shape = (1, 192, 192)
# batch_size = 2
# weights_list = None
# from SYECE.model import ResNeXt
# model_folder = model_root + '/ResNeXt_CBAM_CV_20200814'
# casename_list = []
#
# cv_folder_list = [one for one in IterateCase(model_folder, only_folder=True, verbose=0)]
# cv_pred_list = []
# for cv_index, cv_folder in enumerate(cv_folder_list):
#     model = ResNeXt(3, 2).to(device)
#     if weights_list is None:
#         one_fold_weights_list = [one for one in IterateCase(cv_folder, only_folder=False, verbose=0) if
#                                  one.is_file()]
#         one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
#         weights_path = one_fold_weights_list[-1]
#     else:
#         weights_path = weights_list[cv_index]
#
#     print(weights_path.name)
#     model.load_state_dict(torch.load(str(weights_path)))
#     model.eval()
#     pred_list = []
#     for case in sorted(os.listdir(os.path.join(data_root, 'T2Slice/Test'))):
#         t2, _ = ExtractBlock(np.load((os.path.join(data_root, 'AdcSlice/Test/' + case))), input_shape, [-1, -1, -1])
#         adc, _ = ExtractBlock(np.load((os.path.join(data_root, 'DwiSlice/Test/' + case))), input_shape, [-1, -1, -1])
#         dwi, _ = ExtractBlock(np.load((os.path.join(data_root, 'AdcSlice/Test/' + case))), input_shape, [-1, -1, -1])
#         dismap, _ = ExtractBlock(np.load((os.path.join(data_root, 'DistanceMap/Test/' + case))), input_shape, [-1, -1, -1])
#
#         inputs = MoveTensorsToDevice([torch.tensor(t2[np.newaxis, ...]),
#                                       torch.tensor(adc[np.newaxis, ...]),
#                                       torch.tensor(dwi[np.newaxis, ...]),
#                                       torch.tensor(dismap[np.newaxis, ...].astype(np.float32))], device)
#
#         preds = model(*inputs)[:, 1]
#         pred_list.append((1 - preds).cpu().data.numpy().squeeze().tolist())
#
#         if cv_index == 0:
#             casename_list.append(case[: case.index('_slice')])
#     del model, weights_path
#     plt.hist(pred_list)
#     plt.show()
#     cv_pred_list.append(pred_list)
# case_dict = {}
# final_pred = np.mean(np.array(cv_pred_list), axis=0).tolist()
# plt.hist(final_pred)
# plt.show()
# for idx, name in enumerate(casename_list):
#     case_dict[name] = final_pred[idx]
# print(case_dict)
# import pandas as pd
# pd.DataFrame(casename_list)
# pd.to_csv(r'C:\Users\ZhangYihong\Desktop\JSPH_try.csv', mode='a+')

# suh_clinical_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\SUH_ECE_clinical-report.csv',
#                               encoding='gbk', usecols=['case', '包膜突破'], index_col=['case'])
# suh_label_df = pd.read_csv(r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\label.csv',
#                               encoding='gbk', usecols=['case', 'label'], index_col=['case'])
#
# SUH_path = r'X:\CNNFormatData\ProstateCancerECE\SUH_Dwi1500\AdcSlice'
#
# case_list = os.listdir(SUH_path)
# case_list = [case[:case.index('_-_')] for case in case_list]
# name_list = []
# version_1 = []
# version_clinical = []
# for index in sorted(suh_clinical_df.index):
#     if not index in case_list:
#         continue
#     else:
#         clinical_info = suh_clinical_df.loc[index]
#         suh_info = suh_label_df.loc[index]
#         if int(clinical_info['包膜突破']) != int(suh_info['label']):
#             name_list.append(index)
#             version_1.append(int(suh_info['label']))
#             version_clinical.append(int(clinical_info['包膜突破']))
# df = pd.DataFrame({'case': name_list, 'version 1': version_1, 'version clinical': version_clinical})
# print()
# df.to_csv(r'C:\Users\ZhangYihong\Desktop\different_label_case.csv')


def Check():
    pass


if __name__ == '__main__':

    # target_folder = r'X:\RawData\GS\Prostate of RP 291\Prostate of RP 291'
    # source_folder = r'E:\RP+GS_202005\Prostate of RP 291\Prostate of RP 291'
    # source_case_list = os.listdir(source_folder)
    # target_case_list = os.listdir(target_folder)
    #
    # for case in target_case_list:
    #     # print(case)
    #     # case path
    #     source_path = os.path.join(source_folder, case)
    #     target_path = os.path.join(target_folder, case)
    #
    #     # get target root and the dirs & file in this root
    #     for root_target, dirs_target, files_target in os.walk(target_path, topdown=True):
    #         root_source = os.path.join('E:\RP+GS_202005', root_target.split('X:\RawData\GS\\')[-1])
    #         root_list_source = os.listdir(root_source)
    #         if sorted(dirs_target + files_target) == sorted(root_list_source):
    #             continue
    #         else:
    #             print("{} is missing {} at {}".
    #                   format(case, list(set(root_list_source).difference(set(dirs_target + files_target))),
    #                          root_target.split(target_path)[-1]))
    #             print('*' * 30)
    data = np.load(r'X:\CNNFormatData\ProstateCancerECE\NPYNoDivide\AdcSlice\BI JUN_slice11.npy')
    print(data.shape)
