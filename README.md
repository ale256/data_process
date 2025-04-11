# UltrasoundDatasetBuild数据库创建类使用

## 简介：

    UltrasoundDatasetBuild是一个由wth创建的辅助超声数据库建立的类，用于统一所建立的数据库格式以及存储方式，里面包含了对应数据库自动建立索引、自动保存的功能，可以统一格式规范，避免重复造轮子。

## 使用说明：

依赖环境：

- tqdm
- opencv-python

第一步：导入UltrasoundDatasetBuild类

```python
from utils.ultrasound_dataset_build import UltrasoundDatasetBuild
```

第二步：实例化UltrasoundDatasetBuild类对象

```python
ud = UltrasoundDatasetBuild(ultrasound_dataset_name, save_path, data_type='img', create_user='wth')
```

- :param ultrasound_dataset_name: 数据集名字
- :param save_path: 保存位置的根目录
- :param data_type: img 或 video
- :param create_user: 数据集创建的用户

第三步：初始化保存位置

```python
ud.init_save_folder()
```

第四步：调用UltrasoundDatasetBuild类中的写入数据函数

```python
ud.write_data(*, data, seg: Optional[NDArray[bool]], seg_channel_name: Optional[list],
                classes_dict: Optional[dict], caption: Optional[str], report: Optional[str], 
		box: Optional[list], anatomy: Optional[str], show_seg: bool = False, 
		measurement: Optional[dict], demographic: Optional[dict], biochemical: Optional[dict], 
		original_path: Optional[Union[str, list]], keypoints: Optional[dict], keypoint_names: Optional[list], 
		split: Optional[str], patient_id: str, notes: Optional[str] = None)
 
```

- :param data: 图像或视频，如果是图像，请传入一个npy格式的矩阵（h,w,c）;如果是视频，请传入一个avi格式的视频路径，可以是一个列表，包含多个数据
- :param seg: 分割图像，默认为None，只有在传入的是img数据才会有seg,输入格式为（c,h,w）,保存为npy格式,dtype为bool
- :param seg_channel_name: 分割类别，格式：list ['tumor', ...]，对应seg的通道数
- :param classes_dic: 图像类别，请具体到对应的病种，而不是 'lesion' 这样笼统的类别，输入一个字典，可以包含多个类别
- :param caption: 图像/视频标题，text文本
- :param report: 图像/视频报告，text文本
- :param box: 目标检测框，请传入一个字典，格式 { 类别名 ：[<x_center> <y_center> ],[...],...}， x_center指的是相对于原图的比例，如果传入的是seg图像会自动计算box
- :param anatomy: 图像对应身体的哪一个部位，比如肺部超声，如果数据集没有说明就写default
- :param show_seg: 可视化分割图像转化的目标检测框，用于调试
- :param measurement: 输入一个字典，包含超声测量指标 eg：{"EF": 78.5, "ESV":14.9, "EDV": 69.2}
- :param demographic: 输入一个字典，包含患者基本信息 eg：{"Gender": "male", "Age": 18, "BMI": 20.5, "BloodPressure": 23.4}, 命名请遵循驼峰命名法，value内容小写
- :param biochemical: 输入一个字典，包含患者临床生化指标，如血液检测信息
  eg：{"Scr(male $53\sim106\mu mol/L$,  female:$44\sim97\mu mol/L$)": 88, TC($2.8\sim5.17mmol/L$): 4}, 命名请遵循驼峰命名法，value内容小写
- :param original_path: 原始文件路径，用于追踪数据来源
- :param keypoints: 关键点坐标字典，格式：{"keypoint_name": [x, y], ...}，其中x,y为相对坐标(0-1)，若关键点不存在则为None
- :param keypoint_names: 关键点名称列表，用于保持一致的顺序
- :param split: 数据所属的分割集('train', 'val', 'test'或None)

## 注意事项：

1. write_data函数一次往数据库中写入一个数据，重复调用write_data函数遍历原始数据库即可完成数据库的建立
2. 如果data是图像，请传入一个npy格式的矩阵（h,w,c）；如果data是视频，请传入一个avi格式的视频路径
3. anatomy指的是超声图像对应的是身体哪一个部位，请用英文写入
4. 如果传入分割图像，请传入一个npy格式的矩阵，输入格式为（c,h,w），dtype为bool形，同时需要输入seg_channel_name，即每一个分割通道的名字，格式为list ['tumor', ...]，seg_channel_name[i]对应第i个通道，每个分割通道的名字请自行从原文查找。
5. box: 目标检测框  请传入一个字典，格式 { 类别名 ：[<x_center> <y_center> `<width>` `<height>`],[...],...}， x_center指的是相对于原图的比例，如果传入的是seg图像会自动计算box

   eg："box": [
   {
   "tumor": [
   0.5578291814946619,
   0.31210191082802546,
   0.09074733096085409,
   0.055201698513800426
   ]
   }
   ]
6. 对于没有规定命名的字典，比如measurement，请参看注释里的命名方法
7. 可以参照 [23_breast-ultrasound-images-dataset_process.py](23_breast-ultrasound-images-dataset_process.py) 程序查看数据库建立的具体流程

## 更新日志：

### ！！重要更新

- **Apr 5**: write_data中全部argument改为**必须**输入，如果不需要，请传入None
- **Apr 8 (@WTH)**：取消了sub classes，现在的classes变成了classes_dict，对应一个病例可能有多个分类任务以及分级的情况

### 2025年4月

- **Apr 11**: mixture data type下classes_dict支持list
- **Apr 9**：mixture data type下img_list支持seg_list
- **Apr 8 (@WTH)**：data支持列表输入，需要在定义类的时候把type改成mixture，同时如果输入是列表输入将不支持传分割图，相当于对应一个病人有多个图像的情况
- **Apr 7**：新增patient_id作为write_data函数中必要argument（若无ID信息则输入None)；新增数据集描述栏，使用方法为ud.set_dataset_description("xxx")
- **Apr 6**：fix dataset_info key error; fix draw_bounding_boxes_on_image function（感谢金泽反馈)
- **Apr 6**：新增备注栏，在write_data中可以添加notes(非必要)；以及引入ud.set_dataset_notes函数，可为整个数据集设置备注，若有数据集相关补充信息请放入此栏；新增SubClassesList记录；新增BoxList记录
- **Apr 5**: 新增IncludeSplit记录 - 若原数据集中包含现成train/val/test split，则需依此输入至json文件中
- **Apr 4**: 新增Keypoint Detection Task - 支持关键点检测任务
- **Apr 4**: 新增IncludeMeasurement选项 - 用于记录测量指标
- **Apr 1**：数据集处理成json的时候要保留原文件名
