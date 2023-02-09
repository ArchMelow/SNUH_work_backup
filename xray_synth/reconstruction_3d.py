import numpy as np
import pandas as pd
import pydicom as pdcm
import os
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import fetch_data as FD
from skimage.transform import resize
from scipy.ndimage import rotate


'''
dicom 파일을 열고 metadata 중 slice thickness와 
Pixel Spacing을 가지고 온다.

환자 한 명의 2d slice들을 3d voxel로 재구성해본다. 

'''

def test_func() :
    #ADD TEST FUNC
    return

'''
is_identical()

리스트나 iterable한 자료구조의 모든 원소들이
같은 값을 가지면 true, 아니면 false 리턴한다.

'''

def is_identical(iterable) :
    if any(iter != iterable[0] for iter in iterable) :
        return False
    return True

'''
사용가능한 이미지인지 확인하는 함수.
get_slices()의 일부를 그대로 가져왔다.

'''
def validate_dicom(single_patient_data) :
    # bring all the slices into a single list.
    slices = [pdcm.dcmread(file) for file in single_patient_data['dicom_files']]

    # Check if all slices' ST and PS values are identical to each other.

    st_same = is_identical([s.SliceThickness for s in slices])
    ps_same = is_identical([s.PixelSpacing for s in slices])
    shape_same = is_identical([s.pixel_array.shape for s in slices])

    if not st_same or not ps_same or not shape_same:
        return None

    return True

'''
convert_image()

12 비트 포맷으로 되어 있는 Dicom 이미지를 0~255의 8 비트 포맷으로 바꾼다.
z = x - min(x) / max(x) - min(x)
z = z* 255

인자 : dicom 파일 한 slice 의 pixel array (512 x 512)

'''

def convert_image(slice) :
    tmp = slice.copy()
    #min, max = np.min(tmp), np.max(tmp)
    _tmp = tmp - np.min(tmp)
    # slice 가 아예 모든 픽셀이 검은색 (0) 인 경우에는 연산을 수행하지 않는다.
    if np.max(_tmp) != 0 :
        tmp = _tmp / np.max(_tmp)
    tmp = tmp * 255
    tmp = tmp.astype(np.uint8)
    '''
    print(np.max(tmp))
    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            if tmp[i][j] != 255 :
                tmp[i][j] = 0
    plt.imshow(tmp)
    plt.show()
    '''
    return tmp




'''
get_slices()

특정 환자 (ID) 의 DICOM 파일에 있는 pixel map들(slices)을 불러와서 저장한다.
그리고 세 가지 해부학적 관점 (axial, coronal, sagittal) 에 따라 display 한다.

인자 : fetch_data.py에서 불러온 patient_info dict 의 한 element (환자 한 명의 데이터)
리턴값 : dict (aspects(list), 3d_img_shape(tuple))
img shape는 (512, 512, slice 개수)로 리턴된다.

+ 추가 : rotation 속도를 높이기 위해 이미지의 size를 줄였다. (128, 128)

'''

def get_slices(single_patient_data) :

    # bring all the slices into a single list.
    slices = [pdcm.dcmread(file) for file in single_patient_data['dicom_files']]

    # Check if all slices' ST and PS values are identical to each other.

    st_same = is_identical([s.SliceThickness for s in slices])
    ps_same = is_identical([s.PixelSpacing for s in slices])
    shape_same = is_identical([s.pixel_array.shape for s in slices])


    print(f"Pixel Spacing이 일치 : {ps_same}")
    print(f"Slice Thickness가 일치 : {st_same}")
    print(f"Shape가 일치 : {shape_same}")

    if not st_same or not ps_same or not shape_same:
        print('이 환자의 이미지는 모든 이미지에서 PS 와 ST, Shape가 같지 않습니다.')
        return None

    # print out pixel spacing value (pair) and slice thickness.


    print(f'Slice Thickness 값 : {slices[0].SliceThickness}')
    print(f'Pixel Spacing 값 : {slices[0].PixelSpacing}')
    print(f'Image Shape 값 : {slices[0].pixel_array.shape}')

    # define axes to use 

    ps = slices[0].PixelSpacing # x axis : ps[0], y axis : ps[1]
    st = slices[0].SliceThickness # z axis
    img_shape = slices[0].pixel_array.shape
    #pa = list(img_shape)
    #pa.append(len(slices))
    #print(pa)


    # ax.set_aspect : y/x

    axial_aspect = ps[1]/ps[0]
    sagittal_aspect = ps[1]/st
    coronal_aspect = st/ps[0]

    # construct 3d image
    # from this point, we know that the shape of the slices (pixel maps), st, ps are all the same, so we can safely construct a 3d array from slices.

    img_shape = list(img_shape)
    img_shape.append(len(slices))
    img_3d = np.zeros(img_shape)
    print(img_3d.shape)


    # fill 3d empty array with slices
    # we convert the images to RGB scale (12 bit -> 8 bit images)

    for i, s in enumerate(slices) :
        #pixel_array = convert_image(s.pixel_array)
        img_3d[:, :, i] = s.pixel_array

    # display axial, sagittal, and coronal view by plt.imshow.

    print(f'coronal image : {np.max(img_3d[int(img_shape[0]/2),:,:].T)}')

    '''
    # Cube
    # xy-plane : axial, xz-plane : sagittal, yz-plane : coronal
    ax1 = plt.subplot(1,4,1)
    ax1.set_title('Axial View')
    ax1.imshow(img_3d[:,:,int(img_shape[2]/2)], cmap='gray')
    ax1.set_aspect(axial_aspect)

    ax2 = plt.subplot(1,4,2)
    ax2.set_title('Sagittal View')
    ax2.imshow(img_3d[:,int(img_shape[1]/2),:], cmap='gray')
    ax2.set_aspect(sagittal_aspect)

    ax3 = plt.subplot(1,4,3)
    ax3.set_title('Coronal View')
    ax3.imshow(img_3d[int(img_shape[0]/2),:,:].T, cmap='gray')
    ax3.set_aspect(coronal_aspect)

    #plt.show()
    '''
    
    #resize every images to (512, 512)

    if img_3d[:2] != (64, 64) :
        resize(img_3d, (64, 64, img_shape[2]))

    return dict({'aspects' : [axial_aspect, sagittal_aspect, coronal_aspect], 'img_3d' : img_3d})


if __name__ == "__main__" :
    patient_data = FD.get_data_as_dict() # fetch data from the base directory 'wrist'
    for p in patient_data :
        get_slices(patient_data[p]) # test get_slices 