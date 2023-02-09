"""
1. Read Dicom Data
2. Match Segmentation label with CT
3. Save them into hdf5

"""

# First of all, we need to understand the structure of data.

from tqdm import tqdm
import numpy as np
import pandas as pd
import pydicom as pdcm
import os
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob

# 14개의 레이블을 정의한다. 각각은 손뼈의 항목을 의미한다.

LABELDICT = {
    'ulna':0,
    'radius':1,
    'scaphoid':2,
    'lunate':3,
    'triquetrium':4,
    'hamate':5,
    'capitate':6,
    'trapezoid':7,
    'trapezium':8,
    'pisiform':9,
    '1st metacarpal':10,
    '2nd metacarpal':11,
    '3rd metacarpal':12,
    '4th metacarpal':13,
    '5th metacarpal':14,
}

# 아무것도 없는 환경에서 어떻게 데이터를 뽑아낼까

'''
test_module()

만든 함수를 테스트해보기 위한 함수이다.

'''

def test_module() :
    #return_label_files_as_list()
    #return_patient_info_as_dict('D:\\swkim\\data\\raw_wrist\\wrist\\50716759')
    #DO SOMETHING

    return





'''
return_label_files_as_list()

list 타입으로 3D.Dump 엑셀 파일에서 이용가능한 레이블에 해당하는 nii.gz 파일의 이름을 리턴한다.
이용가능한 레이블은 3D__Lesion_Index 로 엑셀에 저장되어 있는 Index에 해당하는 nii.gz 파일을 의미한다.
나중에 이 nii.gz 파일을 nibabel 라이브러리로 읽어들여서 활용하게 된다.

path_arg : stor/results 까지 이어지는 해당 환자의 3D Dump file에 대한 전체 경로. 예 ) D:\swkim\data\raw_wrist\wrist\18585090\61624_20110125\0002_19700101_000000\stor\results\3D.Dump..
'''

def return_label_files_as_list(path_arg = "D:\\swkim\\data\\raw_wrist\\wrist\\18585090\\61624_20110125\\0002_19700101_000000\\stor\\results\\3DView.Dump.P18585090_20110125_SE2.r.csv") :
    # dump file csv, so read_csv()
    df = pd.read_csv(path_arg)
    # label nii.gz files to use (IMPORTANT : including the path.)

    # excludes the last folder in the directory string.
    base_path_arg = "\\".join(path_arg.split("\\")[:-1])

    # in (i, j), the second element represents the label index.
    # finds valid indices and saves the valid label nii.gz files in a list 'label_files'
    label_indices = [(i, j) for i, j in enumerate(df['3D__Lesion_Index'].values) if df['3D_annotation'][i] in LABELDICT.keys()] 
    label_files = [base_path_arg + f"\\lesionAnnot3D-{str(x[1]).zfill(3)}.nii.gz" for x in label_indices]

    #print(label_files)
    return label_files


'''
return_patient_info_as_dict()

dict 타입으로
{dicom 파일들의 경로들, Label의 정보를 담는 nii.gz 파일의 이름들, dicom 파일 이미지들의 shape}
을 리턴하는 함수.

path_arg : patient id 폴더의 경로
'''

def return_patient_info_as_dict(path_arg) :
    
    # 관련된 Dicom file 들과 label file들을 저장할 list를 각각 만들어 놓는다.
    dicom_files = []; label_files = []

    # used_dirpath는 실제 사용되는 dicom file folder의 경로 (여러개가 있을 수 있다.)
    
    # 관심 있는 patient id
    patient_id = os.path.basename(path_arg)
    len_subdirs = len(glob(os.path.join(path_arg, '*')))
    #print(len_subdirs)

    # if multiple subdirs are there, then pass.
    if len_subdirs > 1 : 
        return None
    
    # 유의미한 patient info 들만을 골라낸다.
    # 첫번째 예외사항 : 11440165 폴더는 레이블 directory인 stor를 가지고 있는 폴더가 44070592_20190116\\0004_20190116_011618 뿐이다.
    elif patient_id == "11440165" :
        used_dirpath = os.path.join(path_arg, '44070592_20190116\\0004_20190116_011618')
        dicom_files = sorted([f for f in glob(os.path.join(used_dirpath, '*.dcm'))])

    # 두번째 예외사항 : 46381976 폴더는 제대로 된 CT영상을 가지고 있으면서 stor 폴더를 가지고 있는 디렉토리가 2277_20160219 폴더의 
    # 0201_20160219_151541 뿐이다.
    elif patient_id == "46381976" :
        used_dirpath = os.path.join(path_arg, '2277_20160219\\0201_20160219_151541')
        dicom_files = sorted([f for f in glob(os.path.join(used_dirpath, '*.dcm'))])

    else :
        used_dirpath = glob(os.path.join(path_arg, '*', '*'))[0]
        #print(used_dirpath)
        dicom_files = sorted([f for f in glob(os.path.join(used_dirpath, '*.dcm'))])
    
    #debugging
    #print('saved dicom files path : ', dicom_files)

    # label nii.gz 파일들을 label_files list에 저장한다.
    label_files = return_label_files_as_list(glob(os.path.join(used_dirpath, 'stor', 'results', '3DView*'))[0])
    
    #debugging
    #print('label files : ', label_files)

    # 마지막으로 모든 dicom file들의 pixel array 들 shape은 서로 같으므로, 첫번째 pixel map의 shape를 전체를 대표하는 shape으로 가져온다.
    assert len(dicom_files) != 0
    dicom_image_shape = pdcm.dcmread(dicom_files[0]).pixel_array.shape

    return dict({
        'dicom_files' : dicom_files,
        'label_files' : label_files,
        'image_shape' : dicom_image_shape
    })

'''
get_data_as_dict() :

dict 타입으로
{해당 환자의 id : {dicom 파일들의 경로들, Label의 정보를 담는 nii.gz 파일의 이름들, dicom 파일 이미지들의 shape}, ...}
를 리턴하는 함수. 

'''

def get_data_as_dict(base_dir = 'D:\\swkim\\data\\raw_wrist\\wrist'):
    
    data_directory = os.path.join(base_dir, '..', 'data')
    
    patient_ids = os.listdir(base_dir) # contains only ids ex) ['13670054', '34245463' ....]
    patient_id_dirs = [os.path.join(base_dir, x) for x in patient_ids] # contains full directory path
    #print(patient_id_dirs)

    patient_info = {}
    # use vanilla for loop, since we can't avoid using nested for loops in dict comprehension.
    for id in tqdm(patient_id_dirs, desc = 'Loading, please wait..') :
        patient_dict = return_patient_info_as_dict(id)
        if patient_dict != None :
            patient_info[os.path.basename(id)] = patient_dict
    
    #patient_info = {os.path.basename(id):return_patient_info_as_dict(id) for id in patient_id_dirs}
    #print(patient_info)

    # label의 개수는 각 dicom 파일마다 보통 15개이다.
    # dict 내용을 요약하는 xlsx 파일을 생성한다.

    stat_info = {id : [len(patient_info[id]['dicom_files']), len(patient_info[id]['label_files']), patient_info[id]['image_shape']] for id in patient_ids}
    #print(stat_info)
    stat_cols = ["NumCTSlices", "NumLabelsNibabel", "Shape"]
    df_patient_stats = pd.DataFrame.from_dict(stat_info, orient = 'index')
    df_patient_stats.columns = stat_cols
    os.makedirs(data_directory,exist_ok= True)
    df_patient_stats.to_excel(data_directory + '\\data.xlsx')

    #print(df_patient_stats)
    # 환자들의 CT 영상 이미지들에서 모든 pixel map 들의 shape가 (512, 512)가 아니기 때문에
    # Data Augmentation을 하는 과정에서 resize()를 해주는 과정은 꼭 필요할 것이다. 

    return patient_info

'''
print_patient_info()

가져온 patient 정보를 일부만 프린트하는 helper 함수.
get_data_as_dict()를 import 해서 쓰고 확인용으로 만든 함수이다.

'''

def print_patient_info(patient_info, print_num = 5):
    
    for i, p in enumerate(patient_info) :
        if i > print_num : break
        info = patient_info[p]
        print('--------------------------------------------------')
        print(f'환자 번호 : {p}')
        print(f"관련된 DICOM file 개수 : {len(info['dicom_files'])}")
        print(f"DICOM 파일 경로 예시 : {info['dicom_files'][0]}")
        print(f"사용되는 레이블 (nii.gz) 파일 개수: {len(info['label_files'])}")
        print(f"레이블 파일 이름 예시 : {info['label_files'][0]}")
        print(f"이미지 shape : {info['image_shape']}")
        print('--------------------------------------------------')
    print('끝')    

if __name__ == '__main__' :
    p_info = get_data_as_dict()
    print_patient_info(p_info)
    

    # 테스트
    test_module()
    