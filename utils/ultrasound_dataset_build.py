import json
import os
import shutil
from datetime import datetime
from typing import Optional, Union

import cv2
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray


class UltrasoundDatasetBuild:
    def __init__(self, ultrasound_dataset_name, save_path, data_type='img', create_user='wth'):
        """

        :param ultrasound_dataset_name: 数据集原始位置
        :param save_path: 保存位置的根目录
        :param data_type: img 或 video
        :param create_user: 数据集创建的用户
        """
        self.save_path = save_path

        assert data_type in ['img', 'video', 'mixture']
        self.DataType = data_type

        self.DatasetName = ultrasound_dataset_name
        self.save_dataset_path = os.path.join(self.save_path, self.DatasetName)
        self.save_index_path = os.path.join(self.save_path, 'index')

        self.save_json_path = os.path.join(self.save_path, 'index', self.DatasetName + ".json")

        # 数据存储路径
        self.data_save_dir = os.path.join(self.save_path, self.DatasetName, data_type)
        self.seg_save_dir = os.path.join(self.save_path, self.DatasetName, 'seg')

        self.write_cnt = 0
        self.sample_cnt = 0

        self.dataset_info = {
            'DatasetName': self.DatasetName,
            'DatasetDescription': None,
            'CreateTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'CreateUser': create_user,
            'DataNum': 0,
            'DataType': data_type,
            'IncludeSeg': False,
            'IncludeClasses': False,
            'IncludeCaption': False,
            'IncludeReport': False,
            'IncludeDemographic': False,
            'IncludeBiochemical': False,
            'IncludeMeasurement': False,
            'IncludeKeypoints': False,
            'IncludeBox': False,
            'IncludeSplit': False,
            'IncludeNotes': False,
            'IncludePatientID': False,
            'SegChannel': 0,
            'AnatomyLocation': [],
            'ClassesDict': {},
            'MeasuresList': [],
            'KeypointsList': [],
            'BoxList': [],
            'Notes': None,
            'PatientIDFormat': None,
            'DataInfo': {},
            
        }

    def merge_dicts_keep_all(self, main_dict, new_dict):
        for key, value in new_dict.items():
            if key in main_dict:
                # 如果键已存在
                if isinstance(main_dict[key], dict) and isinstance(value, dict):
                    # 递归合并子字典
                    self.merge_dicts_keep_all(main_dict[key], value)
                elif isinstance(main_dict[key], list):
                    # 主字典的值是列表
                    if isinstance(value, list):
                        main_dict[key].extend(value)  # 合并列表
                    else:
                        main_dict[key].append(value)  # 追加单个值
                    # 去重（保持顺序）
                    seen = set()
                    main_dict[key] = [x for x in main_dict[key] if not (x in seen or seen.add(x))]
                    main_dict[key] = sorted(main_dict[key])
                else:
                    # 主字典的值不是列表
                    if isinstance(value, list):
                        new_list = [main_dict[key]] + value
                        seen = set()
                        main_dict[key] = [x for x in new_list if not (x in seen or seen.add(x))]
                    else:
                        if main_dict[key] != value:  # 避免重复
                            main_dict[key] = [main_dict[key], value]
            else:
                # 键不存在，直接赋值
                main_dict[key] = value
        return main_dict

    def init_save_folder(self):
        try:
            shutil.rmtree(self.save_dataset_path)
        except:
            pass

        os.makedirs(self.save_dataset_path, exist_ok=True)
        os.makedirs(self.save_index_path, exist_ok=True)
        os.makedirs(self.data_save_dir, exist_ok=True)
        self.write_cnt = 0

    @staticmethod
    def generate_bounding_boxes(segmentation_image, class_name):
        """
        根据分割图像生成目标检测框。

        参数:
        segmentation_image (numpy.ndarray): 分割图像，单通道，目标区域为非零值。
        class_name (str): 目标的类别名。

        返回:
        list: 目标检测框列表，格式为 [[<类别名> <x_center> <y_center> <width> <height>],...]
        """
        # 查找轮廓
        # _, contours, _ = cv2.findContours(segmentation_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 尝试使用兼容方式调用 cv2.findContours
        contour_output = cv2.findContours(segmentation_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 根据返回值的数量决定如何解包
        if len(contour_output) == 3:
            _, contours, _ = contour_output
        else:
            contours, _ = contour_output

        height, width = segmentation_image.shape[:2]
        bounding_boxes = {}

        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 计算中心点
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height

            # 计算相对宽度和高度
            relative_width = w / width
            relative_height = h / height

            # 存储边界框信息
            bounding_boxes[class_name] = [x_center, y_center, relative_width, relative_height]

        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes_on_image(image, boxes):
        """
        将边界框绘制到图像上。

        参数:
        image (numpy.ndarray): 原始图像。
        boxes (dict): 目标检测框字典，格式为 {class_name: [x_center, y_center, width, height]}

        返回:
        numpy.ndarray: 绘制了边界框的图像。
        """
        if boxes is None:
            return image
        
        height, width = image.shape[:2]
        image_with_boxes = image.copy()

        for class_name, box_coords in boxes.items():
            x_center, y_center, relative_width, relative_height = box_coords


            # 将相对坐标转换为绝对坐标
            x = int((x_center - relative_width / 2) * width)
            y = int((y_center - relative_height / 2) * height)
            w = int(relative_width * width)
            h = int(relative_height * height)

            # 绘制边界框
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 绘制类别名
            cv2.putText(image_with_boxes, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image_with_boxes

    @staticmethod
    def analyze_patient_id_format(patient_id, current_format=None):
        """
        Analyzes a patient ID to determine its format and updates format tracking information.
        
        Args:
            patient_id: The patient ID to analyze
            current_format: Current format dictionary, if exists. Should contain keys:
                - description: Text description of ID format
                - min_value: Minimum ID value if numeric
                - max_value: Maximum ID value if numeric
                - pattern: Regex pattern or format string
                - example: Example patient ID
        
        Returns:
            dict: Updated format information dictionary
        """
        if current_format is None:
            current_format = {
                'description': None,
                'min_value': None,
                'max_value': None,
                'pattern': None,
                'example': None
            }
        
        if patient_id is None:
            return current_format
        
        # Initialize format if this is the first ID
        if current_format['description'] is None:
            if isinstance(patient_id, (int, str)):
                if str(patient_id).isdigit():
                    current_format['description'] = 'Numeric ID'
                    current_format['pattern'] = 'Integer'
                    current_format['min_value'] = int(patient_id)
                    current_format['max_value'] = int(patient_id)
                else:
                    current_format['description'] = 'String ID'
                    current_format['pattern'] = 'String'
                
                current_format['example'] = str(patient_id)
        
        # Update numeric ranges if applicable
        elif current_format['description'] == 'Numeric ID' and str(patient_id).isdigit():
            current_format['min_value'] = min(
                current_format['min_value'],
                int(patient_id)
            )
            current_format['max_value'] = max(
                current_format['max_value'],
                int(patient_id)
            )
        
        return current_format

    def write_data(self, *, data, seg: Optional[NDArray[bool]], seg_channel_name: Optional[list],
                classes_dict: Optional[dict], caption: Optional[str], report: Optional[str], box: Optional[list], anatomy: Optional[str], show_seg: bool = False, measurement: Optional[dict], demographic: Optional[dict], biochemical: Optional[dict], original_path: Optional[Union[str, list]], keypoints: Optional[dict], keypoint_names: Optional[list], split: Optional[str], patient_id: str, notes: Optional[str] = None):
        """
        All arguments are keyword arguments, and must be provided.

        :param data: 图像或视频，如果是图像，请传入一个npy格式的矩阵（h,w,c）;如果是视频，请传入一个avi格式的视频路径，可以是一个列表，包含多个数据
        :param seg: 分割图像，默认为None，只有在传入的是img数据才会有seg,输入格式为（c,h,w）,保存为npy格式,dtype为bool
        :param seg_channel_name: 分割类别，格式：list ['tumor', ...]，对应seg的通道数
        :param classes_dict: 图像类别，请具体到对应的病种，而不是 'lesion' 这样笼统的类别，输入一个字典，可以包含多个类别
        :param caption: 图像/视频标题，text文本
        :param report: 图像/视频报告，text文本
        :param box: 目标检测框 请传入一个字典，格式 { 类别名 ：[<x_center> <y_center> ],[...],...}，  x_center指的是相对于原图的比例，如果传入的是seg图像会自动计算box
        :param anatomy: 图像对应身体的哪一个部位，比如肺部超声，如果数据集没有说明就写default
        :param show_seg: 可视化分割图像转化的目标检测框，用于调试
        :param measurement: 输入一个字典，包含超声测量指标 eg：{"EF": 78.5, "ESV":14.9, "EDV": 69.2}
        :param demographic: 输入一个字典，包含患者基本信息 eg：{"Gender": "male", "Age": 18, "BMI": 20.5, "BloodPressure": 23.4}, 命名请遵循驼峰命名法，value内容小写
        :param biochemical: 输入一个字典，包含患者临床生化指标，如血液检测信息
        eg：{"Scr(male $53\sim106\mu mol/L$; female:$44\sim97\mu mol/L$)": 88, TC($2.8\sim5.17mmol/L$): 4}, 命名请遵循驼峰命名法，value内容小写
        :param original_path: Original file path of the data
        :param keypoints: Dictionary of keypoint coordinates, format: 
                        {"keypoint_name": [x, y], ...} where x,y are relative coordinates (0-1)
                        or None if the keypoint is not present
        :param keypoint_names: List of keypoint names to maintain consistent ordering
        :param split: Which split this data belongs to ('train', 'val', 'test', or None)
        :param patient_id: Unique identifier for the patient this data belongs to
        :param notes: 数据集注释
        :return:
        """
        # Check if all required parameters are provided
        required_args = {
            'data': data,
            'seg': seg,
            'seg_channel_name': seg_channel_name,
            'classes': classes_dict,
            'caption': caption,
            'report': report,
            'box': box,
            'anatomy': anatomy,
            'show_seg': show_seg,
            'measurement': measurement,
            'demographic': demographic,
            'biochemical': biochemical,
            'original_path': original_path,
            'keypoints': keypoints,
            'keypoint_names': keypoint_names,
            'split': split,
            'patient_id': patient_id,
        }

        for arg_name, arg_value in required_args.items():
            if arg_value is NotImplemented:  # Using NotImplemented instead of None for validation
                raise ValueError(f"Argument '{arg_name}' is not provided. Please input None if not included in the dataset.")
        

        if seg_channel_name is None and seg is not None:
            seg_channel_name = ['tumor']

        if measurement is not None:
            assert isinstance(measurement, dict)
            self.dataset_info['IncludeMeasurement'] = True
            for key, _ in measurement.items():
                if key not in self.dataset_info['MeasuresList']:
                    self.dataset_info['MeasuresList'].append(key)
                    self.dataset_info['MeasuresList'].sort()

        if classes_dict is not None:
            self.dataset_info['IncludeClasses'] = True
            self.merge_dicts_keep_all(self.dataset_info['ClassesDict'], classes_dict)
            self.dataset_info['ClassesDict'] = {k: self.dataset_info['ClassesDict'][k] for k in sorted(self.dataset_info['ClassesDict'])}

        if keypoint_names is None:
            keypoint_names = []

        if keypoints is not None:
            assert isinstance(keypoints, dict), "Keypoints must be a dictionary"
            self.dataset_info['IncludeKeypoints'] = True
            
            # Validate keypoint format
            for kp_name, kp_data in keypoints.items():
                if kp_data is not None:
                    assert len(kp_data) == 2, f"Keypoint {kp_name} must have [x, y] or be None"
                    assert 0 <= kp_data[0] <= 1, f"Keypoint {kp_name} x coordinate must be between 0 and 1"
                    assert 0 <= kp_data[1] <= 1, f"Keypoint {kp_name} y coordinate must be between 0 and 1"
                
                if kp_name not in self.dataset_info['KeypointsList']:
                    self.dataset_info['KeypointsList'].append(kp_name)
                    self.dataset_info['KeypointsList'].sort()

        if anatomy not in self.dataset_info['AnatomyLocation']:
            self.dataset_info['AnatomyLocation'].append(anatomy)
            self.dataset_info['AnatomyLocation'].sort()

        if caption is not None:
            self.dataset_info['IncludeCaption'] = True

        if report is not None:
            self.dataset_info['IncludeReport'] = True

        if demographic is not None:
            assert isinstance(demographic, dict)
            self.dataset_info['IncludeDemographic'] = True

        if biochemical is not None:
            assert isinstance(biochemical, dict)
            self.dataset_info['IncludeBiochemical'] = True

        # Record split information if provided
        if split is not None:
            assert split in ['train', 'val', 'test'], "Split must be one of: train, val, test"
            self.dataset_info['IncludeSplit'] = True

        if notes is not None:
            self.dataset_info['IncludeNotes'] = True

        if box is not None:
            self.dataset_info['IncludeBox'] = True
            # Add any new box classes to BoxList
            if isinstance(box, dict):
                for box_class in box.keys():
                    if box_class not in self.dataset_info['BoxList']:
                        self.dataset_info['BoxList'].append(box_class)
                        self.dataset_info['BoxList'].sort()
            elif isinstance(box, list):
                for box_dict in box:
                    if isinstance(box_dict, dict):
                        for box_class in box_dict.keys():
                            if box_class not in self.dataset_info['BoxList']:
                                self.dataset_info['BoxList'].append(box_class)
                                self.dataset_info['BoxList'].sort()

        if patient_id is not None:
            self.dataset_info['IncludePatientID'] = True
            # Update PatientIDFormat using the analyzer
            self.dataset_info['PatientIDFormat'] = self.analyze_patient_id_format(
                patient_id, 
                self.dataset_info['PatientIDFormat']
            )

        data_name = 'case%06d'%self.write_cnt
        DataInfo = {
            'anatomy_location': anatomy,
            'split': split,
            'data_path': None,
            'original_path': original_path,
            'seg_path': None,
            'seg_channel_name': seg_channel_name,
            'classes': classes_dict,
            'caption': caption,
            'report': report,
            'box': box,
            'measurement': measurement,
            'keypoints': keypoints,
            'demographic': demographic,
            'biochemical': biochemical,
            'patient_id': patient_id,
            'notes': notes
        }

        if classes_dict is not None:
            self.dataset_info['IncludeClasses'] = True

        if self.DataType == 'img':
            assert isinstance(data, np.ndarray)
            try:
                h, w, c = data.shape
                assert c < h
            except:
                assert data.ndim == 2
            save_data_name = 'case%06d.png' % self.write_cnt
            cv2.imwrite(os.path.join(self.data_save_dir, save_data_name), data)
            DataInfo['data_path'] = os.path.join(self.dataset_info['DatasetName'], self.DataType, save_data_name)

            if seg is not None:
                c, h, w = seg.shape
                assert c < h
                assert seg.dtype == bool
                assert len(seg_channel_name) == c
                self.dataset_info['SegChannel'] = c
                self.dataset_info['IncludeSeg'] = True
                os.makedirs(self.seg_save_dir, exist_ok=True)
                save_seg_name = 'case%06d.npy' % self.write_cnt
                np.save(os.path.join(self.seg_save_dir, save_seg_name), seg)
                DataInfo['seg_path'] = os.path.join(self.dataset_info['DatasetName'], 'seg', save_seg_name)

                if box is None:
                    box_list = []
                    for i, seg_classes in enumerate(seg_channel_name):
                        seg_img = seg[i, :, :]
                        boxes = self.generate_bounding_boxes(seg_img, seg_classes)
                        if show_seg:
                            image_with_boxes = self.draw_bounding_boxes_on_image(data.copy(), boxes)
                            # 显示结果图像
                            cv2.imshow('Image with Bounding Boxes', image_with_boxes)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        box_list.append(boxes)
                    DataInfo['box'] = box_list
            self.write_cnt += 1
        elif self.DataType == 'video':
            assert data.lower().endswith('avi')
            assert seg is None
            save_data_name = 'case%06d.avi' % self.write_cnt
            shutil.copy(data, os.path.join(self.data_save_dir, save_data_name))
            DataInfo['data_path'] = os.path.join(self.dataset_info['DatasetName'], self.DataType, save_data_name)
            self.write_cnt += 1
        elif self.DataType == 'mixture':
            assert isinstance(data, list)
            assert seg is None
            DataInfo['data_path'] = []
            for data_item in data:
                if isinstance(data_item, ndarray):
                    try:
                        h, w, c = data_item.shape
                        assert c < h
                    except:
                        assert data_item.ndim == 2
                    save_data_name = 'case%06d.png' % self.write_cnt
                    cv2.imwrite(os.path.join(self.data_save_dir, save_data_name), data_item)
                    DataInfo['data_path'].append(os.path.join(self.dataset_info['DatasetName'], self.DataType,
                                                         save_data_name))
                elif isinstance(data_item, str):
                    assert data_item.lower().endswith('avi')
                    save_data_name = 'case%06d.avi' % self.write_cnt
                    shutil.copy(data_item, os.path.join(self.data_save_dir, save_data_name))
                    DataInfo['data_path'].append(os.path.join(self.dataset_info['DatasetName'], self.DataType,
                                                         save_data_name))
                self.write_cnt += 1

        self.sample_cnt += 1

        self.dataset_info['DataInfo'][data_name] = DataInfo



    def set_dataset_notes(self, notes):
        self.dataset_info['Notes'] = notes

    def set_dataset_description(self, description):
        self.dataset_info['DatasetDescription'] = description

    def write_json(self):
        self.dataset_info['DataNum'] = self.sample_cnt
        if isinstance(self.dataset_info['DataNum'], dict):
            for key, value in self.dataset_info['DataNum'].items():
                if isinstance(value, (np.int64, np.int32)):
                    self.dataset_info['DataNum'][key] = int(value)
                elif isinstance(value, np.ndarray):
                    self.dataset_info['DataNum'][key] = value.tolist()
        # 以写入模式打开文件，并使用 json.dump() 方法将数据写入文件
        with open(self.save_json_path, 'w') as file:
            # ensure_ascii=False 确保非 ASCII 字符能正确保存
            # indent=4 使输出的 JSON 文件有缩进，更易读
            json.dump(self.dataset_info, file, ensure_ascii=False, indent=4)




if __name__=='__main__':
    ud = UltrasoundDatasetBuild(ultrasound_dataset_name='23.breast-ultrasound-images-dataset',
                                save_path = 'E:\Dataset/ultrasound\process_data/')
    print(ud.DatasetName)
