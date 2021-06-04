import os
import time
import glob
import shutil

from MIP4AIM.Functions.DicomInfo import DicomShareInfo
from MIP4AIM.Dicom2Nii.DataReader import DataReader
from MIP4AIM.Dicom2Nii.Dicom2Nii import ConvertDicom2Nii
from MIP4AIM.Application2Series.ManufactureMatcher import MatcherManager, SeriesCopyer
from MIP4AIM.NiiProcess.DwiProcessor import DwiProcessor
from MIP4AIM.NiiProcess.Registrator import Registrator

from CNNModel.SuccessfulModel.ProstateSegment import ProstateSegmentationTrumpetNet
from CNNModel.SuccessfulModel.CSProstateCancerDetection import CSDetectTrumpetNetWithROI

from MeDIT.Log import CustomerCheck, Eclog

class AutoProcessor:
    def __init__(self, raw_folder, processed_folder, failed_folder, segment_model_folder, detect_model_folder, is_overwrite=False):
        self.raw_folder = raw_folder
        self.process_folder = processed_folder
        self.failed_folder = failed_folder
        self.segment_model_folder = segment_model_folder
        self.detect_model_folder = detect_model_folder
        self.is_overwrite = is_overwrite
        self.dcm2niix_path = r'd:\StandardAlongProgram\MRICron\mricrogl_windows\mricrogl\dcm2niix.exe'

        self.matcher = MatcherManager()
        self.matcher.SetConfig(['t2', 'dwi', 'adc'])

        self.dicom_info = DicomShareInfo()
        self.data_reader = DataReader()

        self.series_copyer = SeriesCopyer()
        self.dwi_processor = DwiProcessor()
        self.registrator = Registrator()

        self.prostate_segmentor = ProstateSegmentationTrumpetNet()
        self.pca_detector = CSDetectTrumpetNetWithROI()


    def DetectProstateCancer(self, case_folder):
        t2_path = os.path.join(case_folder, 't2.nii')
        adc_path = os.path.join(case_folder, 'adc_Reg.nii')
        dwi_path = glob.glob(os.path.join(case_folder, 'dwi_b[0-9]*_Reg.nii'))[0]
        prostate_roi = os.path.join(case_folder, r'ProstateROI_TrumpetNet.nii.gz')
        self.pca_detector.Run(t2_path, adc_path, dwi_path, prostate_roi_image=prostate_roi,
                         store_folder=case_folder)

    def SegmentProstate(self, case_folder):
        t2_path = os.path.join(case_folder, 't2.nii')
        self.prostate_segmentor.Run(t2_path, store_folder=case_folder)

    def RegistrateBySpacing(self, case_folder, target_b_value=1500):
        t2_path = os.path.join(case_folder, 't2.nii')
        adc_path = os.path.join(case_folder, 'adc.nii')
        dwi_path = self.dwi_processor.ExtractSpecificDwiFile(case_folder, target_b_value)

        if dwi_path == '':
            return False, 'No DWI with b close to {}'.format(target_b_value)

        self.registrator.fixed_image = t2_path

        self.registrator.moving_image = adc_path
        try:
            self.registrator.RegistrateBySpacing(store_path=self.registrator.GenerateStorePath(adc_path))
        except:
            return False, 'Align ADC Failed'

        self.registrator.moving_image = dwi_path
        try:
            self.registrator.RegistrateBySpacing(store_path=self.registrator.GenerateStorePath(dwi_path))
        except:
            return False, 'Align DWI Failed'

        return True, ''

    def SeperateDWI(self, case_folder):
        self.dwi_processor.Seperate4DDwiInCaseFolder(case_folder)

    def ExtractSeries(self, case_folder, store_case_folder):
        is_work, source_list, dest_list = self.matcher.MatchOneCase(case_folder)
        if is_work:
            dest_list = self.series_copyer.MakeAbsolutePath(dest_list, store_case_folder)
            self.series_copyer.CopySeries(source_list, dest_list)
            return is_work, '', ''
        else:
            return is_work, source_list, dest_list

    def ConvertDicom2Nii(self, case_folder):
        for root, dirs, files in os.walk(case_folder):
            # it is possible to one series that storing the DICOM
            if len(files) > 0 and len(dirs) == 0:
                if self.dicom_info.IsDICOMFolder(root):
                    ConvertDicom2Nii(root, root + '\\..', dcm2niix_path=self.dcm2niix_path)

    def MoveFilaedCase(self, case):
        if not os.path.exists(os.path.join(self.failed_folder, case)):
            shutil.move(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, case))
        else:
            add_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
            shutil.move(os.path.join(self.raw_folder, case), os.path.join(self.failed_folder, '_{}'.format(add_time)))
        if os.path.exists(os.path.join(self.process_folder, case)):
            shutil.rmtree(os.path.join(self.process_folder, case))

    def IterativeCase(self):
        print('Loading: Segment MyModel')
        self.prostate_segmentor.LoadConfigAndModel(self.segment_model_folder)

        print('Loading: Detection MyModel')
        self.pca_detector.LoadConfigAndModel(self.detect_model_folder)

        self.log = CustomerCheck(os.path.join(self.failed_folder, 'failed_log.csv'), patient=1, data={'State': [], 'Info': []})
        self.eclog = Eclog(os.path.join(self.failed_folder, 'failed_log_details.log')).GetLogger()

        while True:
            for case in sorted(os.listdir(self.raw_folder)):
                print(case, '\n')
                case_folder = os.path.join(self.raw_folder, case)
                if not os.path.isdir(case_folder):
                    continue

                store_case_folder = os.path.join(self.process_folder, case)
                if not os.path.exists(store_case_folder):
                    os.mkdir(store_case_folder)
                else:
                    if not self.is_overwrite:
                        continue

                print('Convert Dicom to Nii: {}'.format(case))
                try:
                    self.ConvertDicom2Nii(case_folder)
                except Exception as e:
                    self.log.AddOne(case, {'State': 'Dicom to Nii failed.', 'Info': e.__str__()})
                    self.eclog.error(e)
                    self.MoveFilaedCase(case)
                    continue

                print('Extract Target Series: {}'.format(case))
                try:
                    is_work, message_one, message_two = self.ExtractSeries(case_folder, store_case_folder)
                    if not is_work:
                        self.log.AddOne(case, {'State': message_one, 'Info': message_two})
                        self.MoveFilaedCase(case)
                        continue
                except Exception as e:
                    self.log.AddOne(case, {'State': 'Series are crashed', 'Info': e.__str__()})
                    self.eclog.error(e)
                    self.MoveFilaedCase(case)
                    continue

                print('Seperate 4D Nii: {}'.format(case))
                try:
                    self.SeperateDWI(store_case_folder)
                except Exception as e:
                    self.log.AddOne(case, {'State': 'Seperate DWI failed.', 'Info': e.__str__()})
                    self.eclog.error(e)
                    self.MoveFilaedCase(case)
                    continue

                print('Registrate Different series: {}'.format(case))
                try:
                    is_work, message = self.RegistrateBySpacing(store_case_folder)
                    if not is_work:
                        self.log.AddOne(case, {'State': 'Registration failed. ', 'Info': message})
                        self.MoveFilaedCase(case)
                        continue
                except Exception as e:
                    self.log.AddOne(case, {'State': 'Registration failed.', 'Info': e.__str__()})
                    self.eclog.error(e)
                    self.MoveFilaedCase(case)
                    continue

                print('Segment Prostate: {}'.format(case))
                try:
                    self.SegmentProstate(store_case_folder)
                except Exception as e:
                    self.log.AddOne(case, {'State': 'Segment prostate failed.', 'Info': e.__str__()})
                    self.eclog.error(e)
                    self.MoveFilaedCase(case)
                    continue

                print('Detect PCa: {}'.format(case))
                try:
                    self.DetectProstateCancer(store_case_folder)
                except Exception as e:
                    self.log.AddOne(case, {'State': 'PCa detection failed.', 'Info': e.__str__()})
                    self.eclog.error(e)
                    self.MoveFilaedCase(case)
                    continue

            self.log.Save()

            print('Sleep........ZZZ.........ZZZZ..........')
            time.sleep(3600)

def main():
    # raw_folder = r'data\temp_dicom'
    # store_folder = r'data\Processed'
    # failed_folder = r'data\Failed'
    raw_folder = r'C:\Users\yangs\Desktop\data\dicom'
    store_folder = r'C:\Users\yangs\Desktop\data\processed'
    failed_folder = r'C:\Users\yangs\Desktop\data\failed'
    segment_model_folder = r'd:\SuccessfulModel\ProstateSegmentTrumpetNet'
    detect_model_folder = r'd:\SuccessfulModel\PCaDetectTrumpetNetBlurryROI1500QA_ZYD_Recheck_V1'
    processor = AutoProcessor(raw_folder, store_folder, failed_folder, segment_model_folder, detect_model_folder, is_overwrite=True)
    processor.IterativeCase()

if __name__ == '__main__':
    main()