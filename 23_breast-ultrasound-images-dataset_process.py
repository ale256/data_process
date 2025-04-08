import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.ultrasound_dataset_build import UltrasoundDatasetBuild

# Dataset information
data_root = '/media/ps/data/Datasets/ultrasound/Dataset_BUSI_with_GT'

# Output path
out_dir = '/media/ps/data/Datasets/ultrasound/bm_pre_json/23'

tasks = 'malignant breast cancer'

ud = UltrasoundDatasetBuild('23.breast-ultrasound-images-dataset', out_dir,
                            data_type='img', create_user='xxx')

classes_list = os.listdir(data_root)

ud.init_save_folder()

cnt = 0
for class_name in classes_list:
    data_path = os.path.join(data_root, class_name)
    if os.path.isdir(data_path):
        data_list = os.listdir(data_path)
        for data_file in tqdm(data_list):
            if data_file.split('.')[-2].split('_')[-1] != 'mask' and data_file.split('.')[-2].split('_')[-1] != '1' and data_file.split('.')[-2].split('_')[-1] != '2':
                seg_name = data_file.split('.')[-2] + '_mask.png'
                original_path = os.path.join(data_path, data_file)
                img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
                seg = cv2.imread(os.path.join(data_path, seg_name), cv2.IMREAD_GRAYSCALE) // 255
                seg = seg.astype(bool)
                if data_file.split('.')[-2] + '_mask_1.png' in data_list:
                    seg1 = cv2.imread(os.path.join(data_path, data_file.split('.')[-2] + '_mask_1.png' ), cv2.IMREAD_GRAYSCALE) // 255
                    seg1 = seg1.astype(bool)
                    seg = seg + seg1
                if data_file.split('.')[-2] + '_mask_2.png' in data_list:
                    seg2 = cv2.imread(os.path.join(data_path, data_file.split('.')[-2] + '_mask_2.png' ), cv2.IMREAD_GRAYSCALE)// 255
                    seg2 = seg2.astype(bool)
                    seg = seg + seg2
                seg = np.expand_dims(seg, 0)

                class_dic = {
                    tasks: class_name,
                }
                
                ud.write_data(
                    data=img,
                    seg=seg,
                    seg_channel_name=['tumor'],
                    classes_dict=class_dic,
                    caption=None,
                    report=None,
                    box=None,
                    anatomy=tasks,
                    show_seg=False,
                    measurement=None,
                    demographic=None,
                    biochemical=None,
                    original_path=original_path,
                    keypoints=None,
                    keypoint_names=None,
                    split=None,
                    patient_id='%05d'%cnt,
                )
                cnt += 1

ud.write_json()




