from collections import defaultdict
import numpy as np
import os
from PIL import Image

# 可选：用于绘图，如果有需求再启用
# import matplotlib.pyplot as plt

# 路径配置
output_dir = "/home/huaxi/Yuheng/sam2/segment-anything-2/notebooks/segmentation_results_obj2"
label_dir = "/home/huaxi/Yuheng/sam2/segment-anything-2/notebooks/videos/mask"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 标签颜色映射
label_colors = {
    (0, 255, 0): 2,
}

def load_label_image(image_path):
    """加载PNG格式的标签图像并转换为class ID矩阵"""
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    class_id_map = np.zeros(image.shape[:2], dtype=np.uint8)

    for color, class_id in label_colors.items():
        mask = np.all(image == color, axis=-1)
        class_id_map[mask] = class_id

    return class_id_map

def calculate_iou_dice(pred_mask, true_mask):
    """计算IOU和Dice系数"""
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0
    dice = 2 * intersection / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0
    return iou, dice

if __name__ == "__main__":

    video_segments = {}  # TODO: 替换为真实数据

    iou_sum = defaultdict(float)
    dice_sum = defaultdict(float)
    count_per_obj = defaultdict(int)

    for out_frame_idx, masks in video_segments.items():
        if out_frame_idx < 5:
            continue

        frame_output_path = os.path.join(output_dir, f"{out_frame_idx:05d}.npy")
        np.save(frame_output_path, masks)
        print(f"Saved segmentation for frame {out_frame_idx} to {frame_output_path}")

        label_image_path = os.path.join(label_dir, f"frame{out_frame_idx:03d}.png")
        if not os.path.exists(label_image_path):
            print(f"Label image not found for frame {out_frame_idx}, skipping.")
            continue

        label_mask = load_label_image(label_image_path)
        obj_ids_in_label = np.unique(label_mask)
        obj_ids_in_label = obj_ids_in_label[obj_ids_in_label != 0]

        if len(obj_ids_in_label) == 0:
            print(f"Frame {out_frame_idx}: 没有找到有效的对象，跳过")
            continue

        for obj_id in obj_ids_in_label:
            true_mask = (label_mask == obj_id)
            pred_mask = masks.get(obj_id, np.zeros_like(true_mask))

            if pred_mask.sum() == 0:
                print(f"Frame {out_frame_idx}, Object {obj_id}: 模型没有分割，IOU = 0, Dice = 0")
                iou = 0
                dice = 0
            else:
                iou, dice = calculate_iou_dice(pred_mask, true_mask)

            iou_sum[obj_id] += iou
            dice_sum[obj_id] += dice
            count_per_obj[obj_id] += 1

            print(f"Frame {out_frame_idx}, Object {obj_id}: IOU = {iou:.4f}, Dice = {dice:.4f}")

    avg_iou_dice_per_obj = {}
    for obj_id in iou_sum.keys():
        if count_per_obj[obj_id] > 0:
            avg_iou = iou_sum[obj_id] / count_per_obj[obj_id]
            avg_dice = dice_sum[obj_id] / count_per_obj[obj_id]
            avg_iou_dice_per_obj[obj_id] = {'avg_iou': avg_iou, 'avg_dice': avg_dice}
            print(f"Object {obj_id}: 平均IOU = {avg_iou:.4f}, 平均Dice = {avg_dice:.4f}")
        else:
            print(f"Object {obj_id}: 没有进行任何分割")
