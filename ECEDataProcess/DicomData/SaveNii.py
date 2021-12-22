import os

import shutil

from MIP4AIM.Application2Series.SeriesMatcher import SeriesStringMatcher

from MeDIT.Log import CustomerCheck, Eclog


class Mymethod:
    def __init__(self, raw_folder, processed_folder, failed_folder):
        self.raw_folder = raw_folder
        self.process_folder = processed_folder
        self.failed_folder = failed_folder

    def GetPath(self, case_folder):
        for root, dirs, files in os.walk(case_folder):
            if len(files) > 3:
                return root, dirs, files
            elif dirs == files == []:
                return root, dirs, files

    def T2Matcher(self, files):
        t2_matcher = SeriesStringMatcher(include_key=['t2'], exclude_key=['roi', 'ROI', 'diff', 'CA', 'ca', 'drawr', 'map', 'DIS2D', 'PosDisp'],
                                         suffex=('.nii'))
        T2_matcher = SeriesStringMatcher(include_key=['T2'], exclude_key=['roi', 'ROI', 'diff', 'CA', 'ca', 'drawr', 'map'],
                                         suffex=('.nii'))

        t2_result = t2_matcher.Match(files) + T2_matcher.Match(files)
        if len(t2_result) > 1:
            if len(t2_matcher.Match(files)) != 0:
                t2_sup_matcher = SeriesStringMatcher(include_key=['tra'])
                t2_result = t2_sup_matcher.Match(t2_result)
            elif len(T2_matcher.Match(files)) != 0:
                T2_sup_matcher = SeriesStringMatcher(include_key=['Ax'])
                t2_result = T2_sup_matcher.Match(t2_result)

        if len(t2_result) == 0:
            raise Exception('Can not find t2 nii data')

        return sorted(t2_result)

    def DWIMatcher(self, files):
        dwi_matcher = SeriesStringMatcher(include_key='dwi', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr'],
                                          suffex=('.nii', '.bval', '.bvec'))
        DwI_matcher = SeriesStringMatcher(include_key='DWI', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr', 'hr'],
                                          suffex=('.nii', '.bval', '.bvec'))

        diff_matcher = SeriesStringMatcher(include_key='diff', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr', 'MC'],
                                           suffex=('.nii', '.bval', '.bvec'))

        DKI_matcher = SeriesStringMatcher(include_key='DKI', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr', 'MC'],
                                          suffex=('.nii', '.bval', '.bvec'))
        dki_matcher = SeriesStringMatcher(include_key='dki', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr', 'MC'],
                                          suffex=('.nii', '.bval', '.bvec'))
        trace_matcher = SeriesStringMatcher(include_key='trace', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr', 'MC'],
                                          suffex=('.nii', '.bval', '.bvec'))

        dwi_result = dwi_matcher.Match(files) + DwI_matcher.Match(files) + \
                     diff_matcher.Match(files) + \
                     DKI_matcher.Match(files) + dki_matcher.Match(files) + \
                     trace_matcher.Match(files)

        if len(dwi_result) == 0:
            DwI_matcher = SeriesStringMatcher(include_key='DWI', exclude_key=['Reg', 'ADC', 'BVAL', 'drawr'],
                                              suffex=('.nii', '.bval', '.bvec'))
            dwi_result = DwI_matcher.Match(files)
            if len(dwi_result) == 0:
                raise Exception('Can not find DWI or DKI data')

        return dwi_result

    def ADCMatcher(self, files):
        adc_matcher = SeriesStringMatcher(include_key=['adc'], exclude_key=['hr', 'drawr', 'Reg', 'roi'], suffex=('.nii'))
        ADC_matcher = SeriesStringMatcher(include_key=['ADC'], exclude_key=['hr', 'drawr', 'Reg', 'roi'], suffex=('.nii'))
        A_D_C_matcher = SeriesStringMatcher(include_key=['Apparent Diffusion Coefficient'], exclude_key=['hr', 'drawr', 'Reg', 'roi'],
                                          suffex=('.nii'))

        adc_result = adc_matcher.Match(files) + ADC_matcher.Match(files) + A_D_C_matcher.Match(files)

        if len(adc_result) == 0:
            ADC_matcher = SeriesStringMatcher(include_key='ADC', exclude_key=['drawr', 'Reg', 'roi'], suffex=('.nii'))
            adc_result = ADC_matcher.Match(files)
            if len(adc_result) == 0:
                raise Exception('Can not find ADC data')
        return adc_result

    def ROIMatcher(self, files):
        roi_matcher = SeriesStringMatcher(include_key='roi', exclude_key=['drawr', 'adc', 'ADC'], suffex=('.nii', '.csv'))
        roi_result = roi_matcher.Match(files)
        if len(roi_result) == 0:
            raise Exception('Can not find ROI data')

        return roi_result

    def CopyFilaedCase(self, case):
        if not os.path.exists(os.path.join(self.failed_folder, case)):
            shutil.copy(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, case))

    def CopyData(self, result, des_folder, case_path, case_class=None):
        original_path = os.path.join(case_path, result)
        if case_class != None:
            des_path = os.path.join(des_folder, case_class)
            shutil.copy(original_path, des_path)
        else:
            des_path = os.path.join(des_folder, result)
            shutil.copy(original_path, des_path)

    def CopyDWIData(self, result_list, des_folder, case_path):
        des_path = r''
        b_value = ['0', '50', '700', '750', '1400', '1500', '2000']
        if len(result_list) == 3:
            for result in result_list:
                if 'DKI' not in result and 'dki' not in result:
                    if '.bval' in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dwi.bval')
                        shutil.copy(original_path, des_path)
                    if '.bvec' in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dwi.bvec')
                        shutil.copy(original_path, des_path)
                    if '.nii' in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dwi.nii')
                        shutil.copy(original_path, des_path)
                if 'DKI' in result or 'dki' in result:
                    if '.bval' in result and 'DWI' or 'dwi' not in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dki.bval')
                        shutil.copy(original_path, des_path)
                    elif '.bvec' in result and 'DWI' or 'dwi' not in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dki.bvec')
                        shutil.copy(original_path, des_path)
                    elif '.nii' in result and 'DWI' or 'dwi' not in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dki.nii')
                        shutil.copy(original_path, des_path)
        elif len(result_list) == 4:
            for result in result_list:
                for b in b_value:
                    if b in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dwi_b'+b+'.nii')
                        shutil.copy(original_path, des_path)
        else:
            for result in result_list:
                original_path = os.path.join(case_path, result)
                des_path = os.path.join(des_folder, result)
                shutil.copy(original_path, des_path)

    def CopyDKIData(self, result_list, des_folder, case_path):
        if len(result_list) == 3:
            for result in result_list:
                if 'DKI' not in result and 'dki' not in result:
                    continue
                if 'DKI' in result or 'dki' in result:
                    print(result_list)
                    if '.nii' in result and 'DWI' not in result and 'dwi' not in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dki.nii')
                        shutil.copy(original_path, des_path)
                    elif '.bval' in result and 'DWI' not in result and 'dwi' not in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dki.bval')
                        shutil.copy(original_path, des_path)
                    elif '.bvec' in result and 'DWI' not in result and 'dwi' not in result:
                        original_path = os.path.join(case_path, result)
                        des_path = os.path.join(des_folder, 'dki.bvec')
                        shutil.copy(original_path, des_path)

    def run(self):

        self.log = CustomerCheck(os.path.join(self.failed_folder, 'failed_log.csv'), patient=1, data={'State': [], 'Info': []})
        self.eclog = Eclog(os.path.join(self.failed_folder, 'failed_log_details.log')).GetLogger()

        for case in sorted(os.listdir(self.raw_folder)):
            des_case_folder = os.path.join(self.process_folder, case)
            if not os.path.exists(des_case_folder):
                os.mkdir(des_case_folder)
            case_folder = os.path.join(self.raw_folder, case)
            case_path, dirs, files = self.GetPath(case_folder)
            if dirs == files == []:
                self.log.AddOne(case, {'State': 'Have no data'})
                print('{} is failed to find data.'.format(case))
                continue

            print('Copy, roi, dwi and adc: {}'.format(case))
            try:
                t2_result = self.T2Matcher(files)
                # copy
                if len(t2_result) == 1:
                    self.CopyData(t2_result[0], des_case_folder, case_path, 't2.nii')
                else:
                    for result in t2_result:
                        self.CopyData(result, des_case_folder, case_path)
                    print('More than one t2', case)
            except Exception as e:
                self.log.AddOne(case, {'State': 'Failed to copy t2 nii.', 'Info': e.__str__()})
                self.eclog.error(e)
                print('Failed to copy t2.')

            try:
                roi_result = self.ROIMatcher(files)
                # copy
                roi_list = ['roi0', 'roi1', 'roi2']
                if len(roi_result) == 2:
                    for roi in roi_result:
                        if '.csv' in roi:
                            self.CopyData(roi, des_case_folder, case_path, 'roi.csv')
                        else:
                            self.CopyData(roi, des_case_folder, case_path, 'roi.nii')
                else:
                    for roi in roi_result:
                        if '.csv' in roi:
                            self.CopyData(roi, des_case_folder, case_path, 'roi.csv')
                        else:
                            for string in roi_list:
                                if string in roi:
                                    self.CopyData(roi, des_case_folder, case_path, string+'.nii')
            except Exception as e:
                self.log.AddOne(case, {'State': 'Failed to copy roi.', 'Info': e.__str__()})
                self.eclog.error(e)
                print('Failed to copy roi.')

            # try:
            #     dwi_result = self.DWIMatcher(files)
            #     # print(dwi_result)
            #     self.CopyDKIData(dwi_result, des_case_folder, case_path)
            # except Exception as e:
            #     self.log.AddOne(case, {'State': 'Failed to copy dwi.', 'Info': e.__str__()})
            #     self.eclog.error(e)
            #     print('Failed to copy dwi.')

            # try:
            #     adc_result = self.ADCMatcher(files)
            #     if len(adc_result) == 1:
            #         self.CopyData(adc_result[0], des_case_folder, case_path, 'adc.nii')
            #     else:
            #         for result in adc_result:
            #             self.CopyData(result, des_case_folder, case_path)
            #         print('More than one adc', case)
            # except Exception as e:
            #     self.log.AddOne(case, {'State': 'Failed to copy adc.', 'Info': e.__str__()})
            #     self.eclog.error(e)
            #     print('Failed to copy adc.')
            #     print()


def main():
    raw_folder = r'C:\Users\ZhangYihong\Desktop\aaaa'
    store_folder = r'C:\Users\ZhangYihong\Desktop\aaaa'
    failed_folder = r'C:\Users\ZhangYihong\Desktop\aaaa'
    processor = Mymethod(raw_folder, store_folder, failed_folder)
    # processor.CopyOneData('CB^chen bin')

    processor.run()

    # case_folder = os.path.join(raw_folder, 'CB^chen bin')
    # case_path, dirs, files = processor.GetPath(case_folder)
    # case_path, dirs, files = processor.GetPath(r'X:\RawData\ProstateCancerECE\PCa-RP\YFG^yang fu gang')
    #
    # result = processor.DWIMatcher(files)
    # print(result)



if __name__ == '__main__':
    # main()
    for case in os.listdir(r'C:\Users\ZhangYihong\Desktop\aaaa'):
        case_folder = os.path.join(r'C:\Users\ZhangYihong\Desktop\aaaa', case)
        if not os.path.isdir(case_folder):
            continue
        else:
            try:
                for root, dirs, files in os.walk(case_folder):
                    data_list = [file for file in files if file.endswith('.nii')]
                    if len(data_list) > 0:
                        [shutil.copyfile(os.path.join(root, data), os.path.join(case_folder, data)) for data in data_list]
            except Exception as e:
                print(e)