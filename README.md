

# UltrasoundDatasetBuild数据库创建类使用

## 简介：

​	UltrasoundDatasetBuild是一个由wth创建的辅助超声数据库建立的类，用于统一所建立的数据库格式以及存储方式，里面包含了对应数据库自动建立索引、自动保存的功能，可以统一格式规范，避免重复造轮子。

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
ud.write_data(self, data, seg=None, seg_channel_name=None, classes=None, sub_classes=None,
                   caption=None, report=None, box=None, anatomy='default', show_seg=False, measurement=None, demographic=None, biochemical=None)
```

- :param data: 图像或视频，如果是图像，请传入一个npy格式的矩阵（h,w,c）;如果是视频，请传入一个avi格式的视频路径

- :param seg: 分割图像，默认为None，只有在传入的是img数据才会有seg,输入格式为（c,h,w）,保存为npy格式,dtype为bool

- :param seg_channel_name: 分割类别，格式：list ['tumor', ...]，对应seg的通道数

- :param classes: 图像类别，请具体到对应的病种，而不是 'lesion' 这样笼统的类别

- :param classes: 图像二级分类，可选

- :param caption: 图像/视频标题，text文本

- :param box: 目标检测框，请传入一个字典，格式 { 类别名 ：[<x_center> <y_center> ],[...],...}， x_center指的是相对于原图的比例，如果传入的是seg图像会自动计算box

- :param tasks: 图像对应身体的哪一个部位，比如肺部超声，如果数据集没有说明就写default

- :param show_seg: 可视化分割图像转化的目标检测框，用于调试

- :param measurement: 输入一个字典，包含超声测量指标 eg：{"EF": 78.5, "ESV":14.9, "EDV": 69.2}

- :param demographic: 输入一个字典，包含患者基本信息 eg：{"Gender": "male", "Age": 18, "BMI": 20.5, "BloodPressure": 23.4}, 命名请遵循驼峰命名法，value内容小写

- :param biochemical: 输入一个字典，包含患者临床生化指标，如血液检测信息
   eg：{"Scr(male $53\sim106\mu mol/L$,  female:$44\sim 97\mu mol/L$)": 88, TC($2.8\sim 5.17mmol/L$): 4}, 命名请遵循驼峰命名法，value内容小写

  



## 注意事项：

1. write_data函数一次往数据库中写入一个数据，重复调用write_data函数遍历原始数据库即可完成数据库的建立

2. 如果data是图像，请传入一个npy格式的矩阵（h,w,c）；如果data是视频，请传入一个avi格式的视频路径

3. tasks指的是超声图像对应的是身体哪一个部位，请用英文写入

4. 如果传入分割图像，请传入一个npy格式的矩阵，输入格式为（c,h,w），dtype为bool形，同时需要输入seg_channel_name，即每一个分割通道的名字，格式为list ['tumor', ...]，seg_channel_name[i]对应第i个通道，每个分割通道的名字请自行从原文查找。

5. box: 目标检测框  请传入一个字典，格式 { 类别名 ：[<x_center> <y_center> <width> <height>],[...],...}， x_center指的是相对于原图的比例，如果传入的是seg图像会自动计算box

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

8. 有问题请联系翁韬涵

9. **新增需求：数据集处理成json的时候要保留原文件名，可以更改ultrasound_dataset_build.py的第221行，或者直接在存储json时新加一项**

   