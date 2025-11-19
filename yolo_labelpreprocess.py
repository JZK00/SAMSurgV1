import os
import json
import random
import shutil

# 类别映射
INSTRUMENT_CLASSES = {
    "剪刀": 0,
    "马里兰双极钳": 1,
    "单钳": 2,
    "针线夹": 3,
    "塑料夹": 4,
    "其他": 5,
    "塑料钳": 6  # 修改重复的标签
}

# 根目录，即包含 'vid1' 到 'vid12' 的目录
root_dir = 'D:\yolov11\miccai'  # 请替换为您的实际根目录路径

# 目标目录，用于存放重新组织后的数据
output_dir = 'D:\yolov11\yolodata'  # 例如，'./output_dataset'

# 训练集和验证集划分比例
train_ratio = 0.8  # 80% 的数据用于训练，20% 用于验证

# 创建目标目录结构
images_train_dir = os.path.join(output_dir, 'images', 'train')
images_val_dir = os.path.join(output_dir, 'images', 'val')
labels_train_dir = os.path.join(output_dir, 'labels', 'train')
labels_val_dir = os.path.join(output_dir, 'labels', 'val')

os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)
os.makedirs(labels_train_dir, exist_ok=True)
os.makedirs(labels_val_dir, exist_ok=True)

def convert_labelme_to_yolo(json_path, image_path, yolo_label_path):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    shapes = data['shapes']
    
    # 获取图像尺寸
    if 'imageWidth' in data and 'imageHeight' in data:
        image_width = data['imageWidth']
        image_height = data['imageHeight']
    else:
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size
    
    yolo_labels = []
    
    for shape in shapes:
        label = shape['label']
        points = shape['points']
        class_id = INSTRUMENT_CLASSES.get(label)
        
        if class_id is None:
            print(f"未找到类别 '{label}' 的映射，跳过该标注。")
            continue
    
        # 提取所有 x 和 y 坐标
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
    
        # 计算边界框的左上角和右下角坐标
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
    
        # 计算中心点和宽高，归一化到 0~1
        center_x = ((x_min + x_max) / 2) / image_width
        center_y = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
    
        # 保留六位小数
        center_x = round(center_x, 6)
        center_y = round(center_y, 6)
        width = round(width, 6)
        height = round(height, 6)
    
        # YOLO 标签格式
        yolo_label = f"{class_id} {center_x} {center_y} {width} {height}"
        yolo_labels.append(yolo_label)
    
    # 将 YOLO 标签写入文件
    with open(yolo_label_path, 'w', encoding='utf-8') as f:
        for yolo_label in yolo_labels:
            f.write(yolo_label + '\n')

# 获取所有的 'vid' 目录
vid_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('vid')]

all_samples = []

for vid in vid_dirs:
    vid_path = os.path.join(root_dir, vid)
    label_dir = os.path.join(vid_path, 'label')
    image_dir = os.path.join(vid_path, 'left_frames')
    
    # 获取所有的 JSON 文件
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        json_path = os.path.join(label_dir, json_file)
        
        # 假设图像文件与 JSON 文件同名，但扩展名为 '.png' 或 '.jpg'
        image_name = json_file.replace('.json', '.png')
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            image_name = json_file.replace('.json', '.jpg')
            image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"图像文件 {image_name} 不存在，跳过该标注。")
            continue
        
        # 将样本添加到列表中，包含 vid 名称
        all_samples.append((vid, json_path, image_path))

print(f"总共有 {len(all_samples)} 个样本。")

# 打乱样本顺序
random.shuffle(all_samples)

# 划分训练集和验证集
train_size = int(len(all_samples) * train_ratio)
train_samples = all_samples[:train_size]
val_samples = all_samples[train_size:]

print(f"训练集包含 {len(train_samples)} 个样本。")
print(f"验证集包含 {len(val_samples)} 个样本。")

# 处理训练集样本
for vid, json_path, image_path in train_samples:
    # 获取文件名（不含路径）
    base_name = os.path.basename(json_path).replace('.json', '')
    
    # 添加 vid 名称作为前缀，确保文件名唯一
    unique_name = f"{vid}_{base_name}"
    
    # 定义目标路径
    target_image_path = os.path.join(images_train_dir, unique_name + '.jpg')  # 统一使用 .jpg
    target_label_path = os.path.join(labels_train_dir, unique_name + '.txt')
    
    # 转换并保存 YOLO 标签
    convert_labelme_to_yolo(json_path, image_path, target_label_path)
    
    # 复制图像文件，转换为 .jpg 格式（如果需要）
    shutil.copy(image_path, target_image_path)

# 处理验证集样本
for vid, json_path, image_path in val_samples:
    # 获取文件名（不含路径）
    base_name = os.path.basename(json_path).replace('.json', '')
    
    # 添加 vid 名称作为前缀，确保文件名唯一
    unique_name = f"{vid}_{base_name}"
    
    # 定义目标路径
    target_image_path = os.path.join(images_val_dir, unique_name + '.jpg')  # 统一使用 .jpg
    target_label_path = os.path.join(labels_val_dir, unique_name + '.txt')
    
    # 转换并保存 YOLO 标签
    convert_labelme_to_yolo(json_path, image_path, target_label_path)
    
    # 复制图像文件，转换为 .jpg 格式（如果需要）
    shutil.copy(image_path, target_image_path)

print("所有文件已处理完毕！")
