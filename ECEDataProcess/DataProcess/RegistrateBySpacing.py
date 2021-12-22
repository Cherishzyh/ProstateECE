import os
from pathlib import Path
import SimpleITK as sitk


class Registrator():
    def __init__(self, fixed_image='', moving_image=''):
        self.__fixed_image = None
        self.__moving_image = None
        self.SetFixedImage(fixed_image)
        self.SetMovingImage(moving_image)

    def SetFixedImage(self, fixed_image):
        if isinstance(fixed_image, Path):
            fixed_image = str(fixed_image)

        if isinstance(fixed_image, str) and fixed_image:
            self.__fixed_image = sitk.ReadImage(fixed_image)
        elif isinstance(fixed_image, sitk.Image):
            self.__fixed_image = fixed_image

    def GetFixedImage(self):
        return self.__fixed_image

    fixed_image = property(GetFixedImage, SetFixedImage)

    def SetMovingImage(self, moving_image):
        if isinstance(moving_image, Path):
            moving_image = str(moving_image)

        if isinstance(moving_image, str) and moving_image:
            self.__moving_image = sitk.ReadImage(moving_image)
        elif isinstance(moving_image, sitk.Image):
            self.__moving_image = moving_image

    def GetMovingImage(self):
        return self.__moving_image

    moving_image = property(GetMovingImage, SetMovingImage)


    def GenerateStorePath(self, moving_image_path):
        moving_image_path = str(moving_image_path)
        if moving_image_path.endswith('.nii.gz'):
            return moving_image_path[:-7] + '_Reg.nii.gz'
        else:
            file_path, ext = os.path.splitext(moving_image_path)
        return file_path + '_Reg' + ext

    def RegistrateBySpacing(self, method=sitk.sitkBSpline, dtype=sitk.sitkFloat32, store_path=''):
        resample_filter = sitk.ResampleImageFilter()

        resample_filter.SetOutputOrigin(self.__fixed_image.GetOrigin())
        resample_filter.SetOutputSpacing(self.__fixed_image.GetSpacing())
        resample_filter.SetSize(self.__fixed_image.GetSize())
        resample_filter.SetOutputDirection(self.__fixed_image.GetDirection())
        resample_filter.SetInterpolator(method)
        resample_filter.SetDefaultPixelValue(0.0)
        resample_filter.SetTransform(sitk.AffineTransform(3))
        resample_filter.SetOutputPixelType(dtype)

        output = resample_filter.Execute(self.__moving_image)

        if store_path:
            sitk.WriteImage(output, store_path)

        return output


def RegistrateBySpacing(case_folder):
    t2_path = os.path.join(case_folder, 't2.nii')
    adc_path = os.path.join(case_folder, 'adc.nii')
    dwi_path = os.path.join(case_folder, 'dwi_b1500.nii')
    registrator = Registrator(t2_path, adc_path)

    try:
        registrator.RegistrateBySpacing(store_path=registrator.GenerateStorePath(adc_path))
    except:
        return False, 'Align ADC Failed'

    registrator = Registrator(t2_path, dwi_path)
    try:
        registrator.RegistrateBySpacing(store_path=registrator.GenerateStorePath(dwi_path))
    except:
        return False, 'Align DWI Failed'

    return True, ''


if __name__ == '__main__':
    # case_folder = r'C:\Users\ZhangYihong\Desktop\try\BAO ZHENG LI'
    # RegistrateBySpacing(r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath')

    case_folder = r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath'
    case_list = ['CSJ^chen shi jie', 'WU XIAO LEI', 'WXZ^wu xi zhong', 'XSJ^xu shou jun']
    for case in case_list:
        case_path = os.path.join(case_folder, case)
        try:
            RegistrateBySpacing(case_path)
        except Exception as e:
            print(case, e)

    #
    # Path(case_folder)

    # import SimpleITK as sitk
    # # for case in os.listdir(r'C:\Users\ZhangYihong\Desktop\aaaa\OriginalPath\WU XIAO LEI'):
    # case_folder = r'C:\Users\ZhangYihong\Desktop\aaaa\aaa\QGZ^qiu guo zhu'
    # print(sitk.ReadImage(os.path.join(case_folder, 't2.nii')).GetSpacing())
    # print(sitk.ReadImage(os.path.join(case_folder, 'dwi.nii')).GetSpacing())
    # print(sitk.ReadImage(os.path.join(case_folder, 'adc.nii')).GetSpacing())
