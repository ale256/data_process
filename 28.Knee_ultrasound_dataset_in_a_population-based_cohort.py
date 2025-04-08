import os
import zipfile
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from utils.ultrasound_dataset_build import UltrasoundDatasetBuild

os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"  # 某些 OpenCV 版本需要
os.environ["OPENCV_IO_DEBUG"] = "0"         # 关闭 OpenCV 调试信息

# 配置路径
data_root = "/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/ultrasound_dataset/benchmark/28.Knee"  # 假设已解压所有ZIP文件
subject_csv = os.path.join(data_root, "data/reference/dataTable.SUBJECT.csv")
image_ref_csv = os.path.join(data_root, "data/reference/dataTable.IMAGE_REF.csv")

# 定义映射字典
usimgt_category = {
    11: "right anterior suprapatellar longitudinal",
    12: "right anterior suprapatellar longitudinal with power Doppler",
    13: "right anterior suprapatellar transverse in 30 degrees flexion",
    14: "right medial longitudinal",
    15: "right lateral longitudinal",
    16: "right anterior suprapatellar transverse in maximal flexion",
    17: "right posterior medial transverse",
    21: "left anterior suprapatellar longitudinal",
    22: "left anterior suprapatellar longitudinal with power Doppler",
    23: "left anterior suprapatellar transverse in 30 degrees flexion",
    24: "left medial longitudinal",
    25: "left lateral longitudinal",
    26: "left anterior suprapatellar transverse in maximal flexion",
    27: "left posterior medial transverse"
}

age_dict = {
            1: 37,
            2: 42,
            3: 47,
            4: 52,
            5: 57,
            6: 62,
            7: 67
 }

# 读取并合并数据
df_subject = pd.read_csv(subject_csv)
df_image = pd.read_csv(image_ref_csv)
merged_df = pd.merge(df_image, df_subject, on="E03SUBJECTID")

# ==================================================================
# 数据集1：图像分类数据集（按图像类型分类）
# ==================================================================
ud_class = UltrasoundDatasetBuild(
    ultrasound_dataset_name="Knee_US_Classification",
    save_path="/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/ultrasound_dataset/benchmark/save_data",
    data_type='img',
    create_user='WTH'
)
ud_class.init_save_folder()

# 遍历所有图像记录
for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    # 跳过无效数据
    if row["E03PASKR"] in [-99, 888] or row["E03PASKL"] in [-99, 888]:
        continue

    # 构建图像路径
    img_path = os.path.join(data_root, "data/image/ultrasound", 'imageArchive.%s'%row["E03USIMGT"], row["E03USIMGF"])
    if not os.path.exists(img_path):
        continue

    # 读取图像并转换为numpy数组
    img = cv2.imread(img_path)
    img_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    # 构建类别字典
    class_name = usimgt_category.get(int(row["E03USIMGT"]), "unknown")
    classes_dict = {"anatomy": class_name}

    # 构建患者信息
    demographic = {
        "Gender": "Female" if row["E03GENDER"] == 2 else "Male",
        "Age": age_dict[row["E03AGE"]]
    }

    # 写入数据
    ud_class.write_data(
        data=img_np,
        seg=None,
        seg_channel_name=None,
        classes_dict=classes_dict,
        caption=None,
        report=None,
        box=None,
        anatomy="knee",
        show_seg=False,
        measurement=None,
        demographic=demographic,
        biochemical=None,
        original_path=img_path,
        keypoints=None,
        keypoint_names=None,
        split=None,
        patient_id=row["E03SUBJECTID"],
        notes=None
    )
# 设置数据集描述
ud_class.set_dataset_description("Knee ultrasound classification dataset by image view type")
ud_class.write_json()
# ==================================================================
# 数据集2：疼痛/KL分级数据集（按患者分类）
# ==================================================================
ud_grade = UltrasoundDatasetBuild(
    ultrasound_dataset_name="Knee_US_Grading",
    save_path="/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/ultrasound_dataset/benchmark/save_data",
    data_type='mixture',
    create_user='WTH'
)
ud_grade.init_save_folder()

# 按患者分组处理
grouped = merged_df.groupby("E03SUBJECTID")

pain_map = {
            "0": "None",
            "1": "Mild",
            "2": "Moderate",
            "3": "Severe",
         }

KL_map = {
            "0": "No OA",
            "1": "Questionable OA",
            "2": "Mild OA",
            "3": "Moderate OA",
            "4": "Severe OA",
            "99": "Total joint replacement",
         }



for patient_id, group in tqdm(grouped, total=len(grouped)):
    # 跳过无效患者
    if (group["E03PASKR"].iloc[0] in [-99, 888] or
            group["E03RADRPAKKL"].iloc[0] in [-99, 888] or
            group["E03RADLPAKKL"].iloc[0] in [-99, 888]):
        continue

    # 分别处理左右膝
    for side in ["right", "left"]:
        # 选择对应侧面的图像
        side_filter = group["E03USIMGD"] == 1 if side == "right" else group["E03USIMGD"] == 2
        side_images = group[side_filter]

        if len(side_images) == 0:
            continue

        # 加载并拼接所有图像
        img_list = []
        img_path_list = []
        for _, img_row in side_images.iterrows():
            img_path = os.path.join(data_root, "data/image/ultrasound", 'imageArchive.%s'%img_row["E03USIMGT"], img_row["E03USIMGF"])
            img_path_list.append(img_path)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img_list.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not img_list:
            continue

        # 构建类别字典
        pain_key = f"Pain"
        kl_key = f"KL"
        classes_dict = {
            pain_key: str(group[f"E03PASKR"].iloc[0] if side == "right" else group[f"E03PASKL"].iloc[0]),
            kl_key: str(group[f"E03RADRPAKKL"].iloc[0] if side == "right" else group[f"E03RADLPAKKL"].iloc[0])
        }
        classes_dict[pain_key] = pain_map[classes_dict[pain_key]]
        classes_dict[kl_key] = KL_map[classes_dict[kl_key]]

        # 构建患者信息
        demographic = {
            "Gender": "Female" if group["E03GENDER"].iloc[0] == 2 else "Male",
            "Age": age_dict[group["E03AGE"].iloc[0]]
        }

        ud_grade.write_data(
            data=img_list,
            seg=None,
            seg_channel_name=None,
            classes_dic=classes_dict,
            caption=f"{side.capitalize()} Knee Images",
            report=None,
            box=None,
            anatomy="knee",
            show_seg=False,
            measurement=None,
            demographic=demographic,
            biochemical=None,
            original_path=img_path_list,
            keypoints=None,
            keypoint_names=None,
            split=None,
            patient_id=patient_id,
            notes=None
        )



ud_grade.set_dataset_description("Knee ultrasound grading dataset with pain and KL grades")

ud_grade.write_json()