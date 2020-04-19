import SimpleITK as sitk
import pydicom
import os


def dicom():
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(data_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    trans_image = sitk.GetImageFromArray(image_array)   ##其他三维数据修改原本的数据，
    trans_image.CopyInformation(image)
    sitk.WriteImage(trans_image, save_path)


def ReadDICOMHead(data_path):
    # name, age, male,
    try:
        ds = pydicom.read_file(data_path)
        print(ds.PatientID)
        print(ds.PatientSex)
        print(ds.PatientAge)

    except Exception as e:
        print(data_path)


def nii(path):
    from MeDIT.SaveAndLoad import LoadNiiData

    image, _, _ = LoadNiiData(path)
    print(image)


def Try():
    raw_folder = r'X:\RawData\ProstateCancerECE\PCa-RP'
    for case in sorted(os.listdir(raw_folder)):
        case_folder = os.path.join(raw_folder, case)
        case_path, dirs, files = GetPath(case_folder)
        if dirs == []:
            continue
        case_path1 = os.path.join(case_path, dirs[0])
        ReadDICOMHead(os.path.join(case_path1, os.listdir(case_path1)[0]))


if __name__ == '__main__':

    data_path = r'X:\\RawData\\ProstateCancerECE\\PCa-RP\\BAO ZI SHENG\\MR\\20180829\\160924\\7382\\1\\1.2.156.112605.14038010223463.20180829080935'
    save_path = r'C:\Users\ZhangYihong\Desktop\DicomHead'
    ReadDICOMHead(data_path)

