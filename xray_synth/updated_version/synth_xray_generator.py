# canvas 사용해서 gui 에 직접적으로 두 앵글 비교할 수 있게
# matplotlib에서 mean 한 img 3d 로 plot, 자유자재로 축 돌릴 수 있게 추가

# +(2023.1.30) 필터 사용해서 사진 품질 개선
# +(2023.2.09) stackoverflow의 게시물, sitk library 이용해서 회전 작업 시간 단축

import PySimpleGUI as sg
# use sitk
import SimpleITK as sitk
# synthetic x-ray generation 테스트
import numpy as np
import pandas as pd
import pydicom as pdcm
from math import cos, sin
import os
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
import fetch_data as FD
import random
import copy
import time
import reconstruction_3d as RE_3D
from skimage.transform import resize
from scipy.ndimage import rotate
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
from matplotlib import cm
from copy import deepcopy
from skimage import img_as_float
from scipy.ndimage import gaussian_filter, median_filter, maximum_filter, minimum_filter
import scipy.interpolate
from PIL import Image as im

matplotlib.use('TkAgg')

sg.theme("DarkBlue3")
sg.set_options(font=("Courier New", 12))


axial_updown_button = [
                    [sg.Button('▲', size=(1, 1), font='Any 7', border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), key='axial_incr')],
                    [sg.Button('▼', size=(1, 1), font='Any 7', border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), key='axial_decr')]
]
sagittal_updown_button = [
                    [sg.Button('▲', size=(1, 1), font='Any 7', border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), key='sagittal_incr')],
                    [sg.Button('▼', size=(1, 1), font='Any 7', border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), key='sagittal_decr')]
]
coronal_updown_button = [
                    [sg.Button('▲', size=(1, 1), font='Any 7', border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), key='coronal_incr')],
                    [sg.Button('▼', size=(1, 1), font='Any 7', border_width=0, button_color=(sg.theme_text_color(), sg.theme_background_color()), key='coronal_decr')]
]

def draw_fig(canvas, figure) :
    if canvas.children :
        for child in canvas.winfo_children() :
            child.destroy()

    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    # forget former plot
    # figure_canvas_agg.get_tk_widget().forget()
    figure_canvas_agg.draw()

    figure_canvas_agg.get_tk_widget().pack(side = 'top', fill = 'both', expand = 1)
    return figure_canvas_agg

def popup(msg) :
    sg.theme('DarkGrey')
    layout = [[sg.Text(msg)]]
    window = sg.Window('Message', layout, no_titlebar = True, finalize = True)
    return window

def MIP(labelArr, axis=1):
    """
    Label has the shape of (num_labels, 512, 512, num_slices)
    """
    if axis == 0:
        max_slice_per_label = labelArr.sum((2, 3)).argmax(-1)
        # labelArrMax = [labelArr[label_idx, slice_idx, :, :] for label_idx, slice_idx in enumerate(max_slice_per_label)]
        labelArrMax = [np.where(labelArr[label_idx].sum(0) > 0, 1, 0) for label_idx in range(labelArr.shape[0])]
    elif axis == 1:
        max_slice_per_label = labelArr.sum((1, 3)).argmax(-1)
        # labelArrMax = [labelArr[label_idx, :, slice_idx, :] for label_idx, slice_idx in enumerate(max_slice_per_label)]
        labelArrMax = [np.where(labelArr[label_idx].sum(1) > 0, 1, 0) for label_idx in range(labelArr.shape[0])]
    elif axis == 2:
        max_slice_per_label = labelArr.sum((1, 2)).argmax(-1)
        labelArrMax = [labelArr[label_idx, ..., slice_idx] for label_idx, slice_idx in enumerate(max_slice_per_label)]
    return np.stack(labelArrMax, 0)


def rescale_volume(image_series, interpolator, new_size, new_spacing) :
    original_phys_size = [(sz - 1) * spc for sz, spc in zip(image_series.GetSize(), image_series.GetSpacing())]
    new_physical_size = [(sz-1)*spc for sz,spc in zip(new_size, new_spacing)]

    scale_tx = sitk.ScaleTransform(image_series.GetDimension(), [ops/nps for nps, ops in zip(new_physical_size, original_phys_size)])
    rigid_tx = sitk.Euler3DTransform()
    rigid_tx.SetMatrix(image_series.GetDirection())
    rigid_tx.SetTranslation(image_series.GetOrigin())
    #accomodate for non identity direction cosine matrix, ensure that scaling is
    #along the image axes and not the canonical ones.
    tx = sitk.CompositeTransform([rigid_tx, scale_tx, rigid_tx.GetInverse()])

    #output image has the same origin and direction as the input, but is a
    #scaled version of the content
    return sitk.Resample(image_series,
                         size = new_size,
                         transform = tx,
                         interpolator = interpolator,
                         outputOrigin = image_series.GetOrigin(),
                         outputDirection = image_series.GetDirection(),
                         outputSpacing = new_spacing)


# from https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri

def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R

def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def rotation3d(image, theta_x, theta_y, theta_z, show=False):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image (np 3d array)
    """
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform()
    print(euler_transform.GetMatrix())
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)

    # About each axis (x, y, z), the rotation angles are different.
    # So we need to apply three different transformations to the image.

    direction = image.GetDirection()
    axis_angle_x = (direction[0], direction[3], direction[6], theta_x)
    axis_angle_y = (direction[1], direction[4], direction[7], theta_y)
    axis_angle_z = (direction[2], direction[5], direction[8], theta_z)
    
    # order of the rotation would be x -> y -> z, so we need to reverse
    # the order of the transformations.
    
    np_rot_mat = matrix_from_axis_angle(axis_angle_z)
    np_rot_mat = np.dot(np_rot_mat, matrix_from_axis_angle(axis_angle_y))
    np_rot_mat = np.dot(np_rot_mat, matrix_from_axis_angle(axis_angle_x))
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_image = resample(image, euler_transform)
    if show:
        slice_num = int(input("Enter the index of the slice you would like to see"))
        plt.imshow(sitk.GetArrayFromImage(resampled_image)[slice_num])
        plt.show()
    return sitk.GetArrayFromImage(resampled_image)

'''

Main Program

'''

if __name__ == '__main__' :

    former_rotation = dict()
    cwd = os.getcwd()

    canvas_column = sg.Column(
            [
                [sg.Canvas(key = '__CANVAS__', expand_x= True, expand_y = True)]
            ]
    )

    # empty dict and list
    patient_data = dict()
    patient_ids = []

    layout = [
        [
        sg.Column(
            [
                [sg.Text('원하는 환자번호 폴더 선택 :')],
                [sg.InputText(key = '-FILE_PATH-', size = (20, 1)),
                    sg.FolderBrowse(initial_folder=cwd)
                ],
                [sg.Button('폴더 선택')],
                [sg.Text('환자 ID를 선택하세요 : ', size = (20, 1), expand_x= True, expand_y = True)],
                [sg.Text('선택 데이터의 Slice 수 : ', size = (20, 1)), sg.Input(key = 'num_slices', disabled = True, use_readonly_for_disable= False, size = (4,1))],
                [sg.Listbox(enable_events = True, values = [], key = '_SELECTED_', size = (15, 10), select_mode=sg.SELECT_MODE_EXTENDED, horizontal_scroll = True, expand_x = True, expand_y = True)],
                [sg.Text('각 구도에서의 회전 각도를 입력 :', expand_x= True, expand_y = True)],
                [sg.Text('Axial 각도', expand_x= True, expand_y = True), sg.Input(key = 'axial_angle', size = (10,10), expand_x = True, expand_y = True, default_text = 0), sg.Column(axial_updown_button)],
                [sg.Text('Sagittal 각도', expand_x= True, expand_y = True), sg.Input(key = 'sagittal_angle', size = (10,10), expand_x = True, expand_y = True, default_text = 0), sg.Column(sagittal_updown_button)],
                [sg.Text('Coronal 각도', expand_x= True, expand_y = True), sg.Input(key = 'coronal_angle', size = (10,10), expand_x = True, expand_y = True, default_text= 0), sg.Column(coronal_updown_button)],
                [sg.Button('설정 완료', expand_x= True, expand_y = True), sg.Button('값 초기화', expand_x= True, expand_y = True)],
                [sg.Button('생성된 현재 이미지와 각도 저장하기', expand_x = True, expand_y = True)]
            ])
        , canvas_column
        ]
    ]

    window = sg.Window('Synthetic X-Ray Generator', layout, no_titlebar=False, finalize = True, resizable= True)
    window.Maximize()
    window.set_min_size((900, 450))
    canvas_column.expand(True, True)
    window.bind('<Configure>', "Configure")

    while True:
        event, values = window.read()

        if event == '폴더 선택' :
            if not values['-FILE_PATH-'] :
                sg.Popup('경로를 올바르게 선택해 주세요.')
                continue
            
            patient_data = FD.get_data_as_dict(values['-FILE_PATH-'])

            if not patient_data :
                sg.Popup('사용할 수 없는 데이터입니다. Slice 수가 600 보다 많거나, 아예 발견되지 않는 파일입니다.')
                continue

            patient_ids = [p for p in patient_data] # pick only valid data
            window['_SELECTED_'].update(patient_ids)

            print(patient_data)

            dicom_file_dir = '\\'.join(patient_data[patient_ids[0]]['dicom_files'][0].split('\\')[:-1])

            print(dicom_file_dir)

        if event == sg.WINDOW_CLOSED:
            break
        if event == '생성된 현재 이미지와 각도 저장하기' and former_rotation :
            #folder_name = sg.popup_get_folder('This msg will not be displayed.', no_window = True)
            folder_name = dicom_file_dir + '\\DRR'
            if not os.path.exists(folder_name) :
                os.mkdir(folder_name)

            ap_or_lat = sg.PopupOKCancel('이미지를 AP로 저장하고 싶으시면 OK를,\n Lateral로 저장하고 싶으시면 Cancel을 눌러주세요.')

            # Presses Cancel
            if ap_or_lat == None :
                continue

            proceed_warning = sg.PopupOKCancel('**경고**\nOK를 누르시면 전에 있던 이미지가 덮어씌워집니다. 계속하시겠습니까?')

            if proceed_warning in ['Cancel', None] :
                continue 

            img_name = 'drr_lateral.png' if ap_or_lat == 'Cancel' else 'drr_ap.png'

            file_postfix = 'lat' if ap_or_lat == 'Cancel' else 'ap' 

            read_img = im.fromarray(current_xray_gen[0])
            read_img.save(f"{folder_name}\\{img_name}")

            data_to_write = [
                f"Axial : {float(values['axial_angle'])}\n",
                f"Sagittal : {float(values['sagittal_angle'])}\n",
                f"Coronal : {float(values['coronal_angle'])}"
            ]

            with open(f"{folder_name}\\angles_{file_postfix}.txt", 'w') as file :
                file.writelines(data_to_write)

            #read_img.save(f"{folder_name}\\{values['_SELECTED_'][0]}.png")
            sg.Popup(f"{folder_name}에 이미지 파일과 각도 저장 파일이 저장되었습니다.")
        if event == '설정 완료' :
            window['설정 완료'].update(disabled=True)
            if values['_SELECTED_'] == [] :
                sg.Popup('ID를 선택하지 않았습니다. ID를 선택 후 진행해 주세요.')
                window['설정 완료'].update(disabled= False) # disable multiple clicks until the completion of a process.
                continue

            print(values)

            # Load nii.gz label files

            #print(patient_data)

            '''
            for label_file in patient_data[values['_SELECTED_'][0]]['label_files'] :
                print(f'shape of the label read : {nib.load(label_file).get_fdata().shape}')
            '''

            popup_choice = sg.PopupOKCancel('새로운 회전 방법을 시도합니까?')
            # Fix popup_choice to the new rotation method, as we don't use the second useless method for now.
            #popup_choice = 'OK'

            # define rotation angles.
            
            if (values['axial_angle'], values['sagittal_angle'], values['coronal_angle']) == ("", "", "") :
                theta_ax, theta_sg, theta_cor = 0, 0, 0
            else :
                theta_ax, theta_sg, theta_cor = map(float, (values['axial_angle'], values['sagittal_angle'], values['coronal_angle']))


            if popup_choice == None :
                window['설정 완료'].update(disabled= False)
                continue

            elif popup_choice == 'Cancel' :
                
                labels_arr = list()

                for label_file in patient_data[values['_SELECTED_'][0]]['label_files'] :
                    _label_ = nib.load(label_file).get_fdata()
                    _label_ = np.flip(_label_, axis=2)
                    _label_ = np.rot90(np.flip(_label_, axis=0), 3)
                    labels_arr.append(np.where(_label_ != 0, 1, 0))


                if len(labels_arr) > 0 :
                    temp = np.stack(labels_arr, 0).astype(int)
                    label_used_ap = MIP(temp, axis = 0)
                    #label_used_lateral = MIP(temp, axis = 1)
                    
                # There are no labels found.
                else :
                    sg.Popup('레이블 파일이 없습니다. 그대로 진행합니다.')
                    label_used_ap = np.zeros((512, 512)) # empty image array
                
                
                get_slice_start = time.time()
                ret = RE_3D.get_slices(patient_data[values['_SELECTED_'][0]])
                get_slice_end = time.time()
                print(f'get_slice() ended at {get_slice_end - get_slice_start : .5f}')


                if ret is None :
                    sg.Popup('사용할 수 없는 데이터입니다.')
                    window['설정 완료'].update(disabled= False) # disable multiple clicks until the completion of a process.
                    continue
                
                # First, bring img_3d (CT slices stacked z-wards)

                # axial : xy plane, coronal : yz plane, sagittal : xz plane

                img_3d = ret['img_3d']

                print('rotate start!')
                rotate_start_time = time.time()

                # Rotation Angle이 모두 0일 때는 회전을 할 필요가 없으므로 아래 내용을 pass.

                if not theta_ax and not theta_cor and not theta_sg :
                    img_cor = deepcopy(img_3d)
                    #rotating_popup = popup('생성 중...')
                    
                else :
                    img_ax = rotate(img_3d, angle = theta_ax, axes = (0, 1), reshape = True)
                    img_sag = rotate(img_ax, angle = theta_sg, axes = (0,2), reshape = True)
                    img_cor = rotate(img_sag, angle = theta_cor, axes = (1,2), reshape = True)

                rotate_end_time = time.time()
                print(f'rotate elapsed time : {rotate_end_time-rotate_start_time : .5f}')
                window['설정 완료'].update(disabled= False)

            else :
                print('Second rotation method.')
            
                labels_arr = list()

                for label_file in patient_data[values['_SELECTED_'][0]]['label_files'] :
                    _label_ = nib.load(label_file).get_fdata()
                    _label_ = np.flip(_label_, axis=2)
                    _label_ = np.rot90(np.flip(_label_, axis=0), 3)
                    labels_arr.append(np.where(_label_ != 0, 1, 0))


                if len(labels_arr) > 0 :
                    temp = np.stack(labels_arr, 0).astype(int)
                    label_used_ap = MIP(temp, axis = 0)
                    #label_used_lateral = MIP(temp, axis = 1)
                    
                # There are no labels found.
                else :
                    sg.Popup('레이블 파일이 없습니다. 그대로 진행합니다.')
                    label_used_ap = np.zeros((512, 512)) # empty image array

                



                rotate_2nd_start = time.time()
                
                # Image shape requirement for the sitk reader is (512, 512).
                # Instead of resizing them, we prompt user to resize them for themselves.

                if patient_data[values['_SELECTED_'][0]]['image_shape'] != (512, 512) : 
                    sg.Popup('Rotation Fail :\n이 회전 방법과 호환되는 DICOM 이미지 크기는 512 x 512 입니다.\n다른 환자번호로 시도하시거나 해당 환자 DICOM 파일들을 512 x 512로 변환 후 사용해주세요.')
                    continue

                try :
                    img_reader = sitk.ImageSeriesReader()
                    print(patient_data[values['_SELECTED_'][0]]['dicom_files'][0].split('\\')[:-1])
                    # 가독성은 매우 떨어지지만 추후에 수정.
                    #print(dicom_dirname)
                    img_reader.SetFileNames(patient_data[values['_SELECTED_'][0]]['dicom_files'])

                    sitk_img_3d = img_reader.Execute()

                except RuntimeError as e :
                    sg.Popup(f'이미지 로드에 실패했습니다 :\nException 내역 : {e}')
                    window['설정 완료'].update(disabled= False)
                    continue 

                sitk_img_size = sitk_img_3d.GetSize()
                sitk_img_spacing = sitk_img_3d.GetSpacing()
                print(f"Loaded 3D img shape : {sitk_img_size}")

                #sitk_img_3d = rescale_volume(sitk_img_3d, sitk.sitkNearestNeighbor, (sitk_img_size[2], sitk_img_size[2], sitk_img_size[2]), sitk_img_spacing)

                img_cor = rotation3d(image = sitk_img_3d, 
                                          theta_x = theta_cor, 
                                          theta_y = theta_sg,
                                          theta_z = theta_ax, 
                                          show = False).transpose(1, 2, 0) # transpose 3d image to (x, y, z)

                rotate_2nd_end = time.time()
                print(f'rotate 2nd elapsed time : {rotate_2nd_end - rotate_2nd_start : .5f}')
                
            '''
            center_xyz = [0,0,0]
            rot_mat = getRodriguesMatrix(image = img_3d, x_center = 0, y_center = 0, z_center = 0, axis = (0,0,1), theta = )
            '''
            


            # final image : img_cor
            print(f'shape of the final x-ray image : {img_cor.shape}')

            # method1: mean method

            #scaled_size = 512 if 512 > int(values['num_slices']) else int(values['num_slices'])
            scaled_size = int(values['num_slices'])

            xray1 = resize(np.mean(img_cor, axis = 0), (scaled_size, scaled_size))

            xray2 = resize(np.mean(img_cor, axis = 1), (scaled_size, scaled_size))

            '''
            figs, axs = plt.subplots(1,2)
            axs[0].imshow(xray1, cmap = 'gray')
            axs[1].imshow(xray2, cmap = 'gray')
            plt.show()
            '''

            # apply filters on the output X-Rays

            radius = 5; amount = 2

            # test out different filters.
            xray_test = [RE_3D.convert_image(xray1), RE_3D.convert_image(xray2)]
            xray_test0, xray_test1 = img_as_float(xray_test[0]), img_as_float(xray_test[1])
            

            mask1 = gaussian_filter(xray_test0, sigma = radius)

            mask2 = gaussian_filter(xray_test1, sigma = radius)


            #mask1 = median_filter(xray_test0, size = 20)

            #mask2 = median_filter(xray_test1, size = 20)


            #print(np.array(masks).shape)

            current_xray_gen = [(np.clip(x + (x- m) * amount, 0., 1.)*255).astype(np.uint8) for x, m in zip([xray_test0, xray_test1], [mask1, mask2])]
            current_label = resize(np.sum(label_used_ap, axis = 0), (scaled_size, scaled_size))


            '''
            figs, axes = plt.subplots(2,4)
            axes = np.ravel(axes)
            for i, m in enumerate(current_xray_gen) :
                axes[i].imshow(m, cmap = 'gray')
        
            plt.show()
            '''
            

            #xray3 = resize(np.mean(img_cor, axis = 2), (512,512))

            print(xray1.shape, xray2.shape)

            # If no data have been rotated before, OR if we have to display a different patient's x-ray, show only the current x-ray image.
            if not former_rotation or former_rotation['patient_id'] != values['_SELECTED_'][0] :
                
                fig = plt.figure()
                ax = fig.subplots(ncols = 3)

                fig.suptitle(f"X-Ray AP, Lateral Obtained By\nAxial : {float(values['axial_angle'])}, Sagittal : {float(values['sagittal_angle'])}, Coronal : {float(values['coronal_angle'])}\nRotated Counter-clockwise", fontsize = 16)
                
                ax[0].imshow(current_xray_gen[0], cmap = 'gray')
                ax[0].set_title('AP With Filter Applied (Gaussian)')

                ax[1].imshow(current_xray_gen[1], cmap = 'gray')
                ax[1].set_title('Lateral With Filter Applied (Gaussian)')

                ax[2].imshow(current_label, cmap = 'gray')
                ax[2].set_title('Label')

                fig_canvas_agg = draw_fig(window['__CANVAS__'].TKCanvas, fig)
                plt.close()

            else :
                fig, ax = plt.subplots(nrows = 2, ncols = 3)
                fig.suptitle(f"X-Ray AP, Lateral Obtained By\nBEFORE Axial : {float(former_rotation['axial_angle'])}, Sagittal : {float(former_rotation['sagittal_angle'])}, Coronal : {float(former_rotation['coronal_angle'])}\nAFTER Axial : {float(values['axial_angle'])}, Sagittal : {float(values['sagittal_angle'])}, Coronal : {float(values['coronal_angle'])}\nRotated Counter-clockwise", fontsize = 16)
                
                

                print(f"For Comparison (Axial : {former_rotation['axial_angle']}, Sagittal : {former_rotation['sagittal_angle']}, Coronal : {former_rotation['coronal_angle']})")
                ax = np.ravel(ax)
                for i in range(6) :
                    #ax[i].axis('off')
                    if (i // 2) % 2 == 0 :
                        ax[i].set_title('AP With Filter Applied (Gaussian)')
                    elif (i // 2) % 2 == 1 :
                        ax[i].set_title('Lateral With Filter Applied (Gaussian)')
                    else :
                        ax[i].set_title('Label')

                    if i == 0 or i == 1 :
                        ax[i].imshow(former_rotation['xray'][i], cmap = 'gray')
                    elif i == 2 :
                        ax[i].imshow(former_rotation['label_ap'], cmap = 'gray')
                    elif i == 3 or i == 4 :
                        ax[i].imshow(current_xray_gen[i-3], cmap = 'gray')
                    elif i == 5 :
                        ax[i].imshow(current_label, cmap = 'gray')

                fig_canvas_agg = draw_fig(window['__CANVAS__'].TKCanvas, fig)
                plt.close()

                
            former_rotation['xray'] = deepcopy(current_xray_gen) 
            former_rotation['axial_angle'] = float(values['axial_angle'])
            former_rotation['sagittal_angle'] = float(values['sagittal_angle'])
            former_rotation['coronal_angle'] = float(values['coronal_angle'])
            former_rotation['patient_id'] = values['_SELECTED_'][0]
            former_rotation['label_ap'] = deepcopy(current_label)

            window['설정 완료'].update(disabled= False) # disable multiple clicks until the completion of a process.
            
            # Capture Function copied from https://www.engineerknow.com/2022/12/crop-image-simple-app-using-cv2-numpy.html
            
            crop_ok = sg.PopupOKCancel('최종 이미지에 여백이 있을 경우 OK를 누르시고, 여백을 제외한 영역을 직접 선택하신 후 C (Capture)키를 눌러주세요.\n여백이 없으면 Cancel을 눌러주세요.')
            
            if crop_ok not in ['Cancel', None] :
                xray_input = current_xray_gen[0] # view first x-ray
                img_dup = np.copy(xray_input)
                mouse_pressed = False
                starting_x=starting_y=ending_x=ending_y= -1
                
                def mousebutton(event, x, y, flags, param):
                    global img_dup , starting_x, starting_y, ending_x, ending_y, mouse_pressed
                    #if left mouse button is pressed then takes the cursor position at starting_x and starting_y 
                    if event == cv2.EVENT_LBUTTONDOWN:
                        mouse_pressed= True
                        starting_x,starting_y=x,y
                        img_dup= np.copy(param)
                
                    elif event == cv2.EVENT_MOUSEMOVE:
                        if mouse_pressed:
                            img_dup = np.copy(param)
                            cv2.rectangle(img_dup,(starting_x,starting_y),(x,y),(124,124,0),2)
                    #final position of rectange if left mouse button is up then takes the cursor position at ending_x and ending_y 
                    elif event == cv2.EVENT_LBUTTONUP:
                        mouse_pressed=False
                        ending_x,ending_y= x,y
                        
                cv2.namedWindow('image')
                cv2.setMouseCallback('image',mousebutton, xray_input) #xray_input as param
                while True:
                    cv2.imshow('image',img_dup)
                    k= cv2.waitKey(1)
                    if k==ord('c'):
                        #remove these condition and try to play weird output will give you idea why its done
                        if starting_y>ending_y:
                            starting_y,ending_y= ending_y,starting_y
                        if starting_x>ending_x:
                            starting_x,ending_x= ending_x,starting_x
                        if ending_y-starting_y>1  and ending_x-starting_x>0:
                            ch = sg.PopupOKCancel('이전에 저장된 사진에 덮어씌워집니다. 계속하시겠습니까?')
                            if ch not in [None, 'Cancel'] :
                                lat_or_ap = sg.PopupOKCancel('Lateral 파일로 저장하면 OK, AP로 저장하면 Cancel을 눌러주세요.')
                                if lat_or_ap == None :
                                    continue
                                folder_name = dicom_file_dir + '\\DRR'
                                if not os.path.exists(folder_name) :
                                    os.mkdir(folder_name)
                                img_name = 'drr_ap.png' if lat_or_ap == 'Cancel' else 'drr_lateral.png'  
                                image = xray_input[starting_y:ending_y,starting_x:ending_x]
                                img_dup= np.copy(image)
                                
                                
                                image = (resize(image, (512, 512))*255).astype(np.uint8)
                                
                                plt.imshow(image, cmap = 'gray')
                                plt.show()
                                
                                
                                read_img = im.fromarray(image) # resize to 512 x 512
                                #read_img = read_img.convert("L") # To grayscale non-float values array
                                print(folder_name, img_name)
                                read_img.save(folder_name + '\\' + img_name) 
                                sg.Popup('저장을 완료했습니다. 캡처를 마치시려면 뷰어에서 키보드의 \'Q\'키를 눌러주세요.')
                                
                        elif k == 27:
                            break
                    elif k == ord('q') :
                        break
                cv2.destroyAllWindows()

                


        if event == 'axial_incr' :
            axial_val = float(values['axial_angle'])
            window['axial_angle'].update(axial_val + 10)
            window['axial_angle'].widget.xview_moveto(1)

        elif event == 'axial_decr' :
            axial_val = float(values['axial_angle'])
            window['axial_angle'].update(axial_val - 10)
            window['axial_angle'].widget.xview_moveto(1)

        elif event == 'sagittal_incr' :
            sagittal_val = float(values['sagittal_angle'])
            window['sagittal_angle'].update(sagittal_val + 10)
            window['sagittal_angle'].widget.xview_moveto(1)

        elif event == 'sagittal_decr' :
            sagittal_val = float(values['sagittal_angle'])
            window['sagittal_angle'].update(sagittal_val - 10)
            window['sagittal_angle'].widget.xview_moveto(1)

        elif event == 'coronal_incr' :
            coronal_val = float(values['coronal_angle'])
            window['coronal_angle'].update(coronal_val + 10)
            window['coronal_angle'].widget.xview_moveto(1)

        elif event == 'coronal_decr' :
            coronal_val = float(values['coronal_angle'])
            window['coronal_angle'].update(coronal_val - 10)
            window['coronal_angle'].widget.xview_moveto(1)    

        elif event == '값 초기화' :
            window['axial_angle'].update(0)
            window['axial_angle'].widget.xview_moveto(1)
            window['sagittal_angle'].update(0)
            window['sagittal_angle'].widget.xview_moveto(1)
            window['coronal_angle'].update(0)
            window['coronal_angle'].widget.xview_moveto(1)

        #elif event == '확대 모드' :
            #plt.show()
        
        elif event == '_SELECTED_' :
            window['num_slices'].update(len(patient_data[values['_SELECTED_'][0]]['dicom_files']))

        #elif event == '종료' :
            #break

        print(event, values)

    window.close()
