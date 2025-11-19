import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ========== 自定义依赖项 (需要你提前定义或导入) ==========
# predictor: Segment-Anything 推理器
# inference_state: 当前推理状态
# frame_names: 所有帧图像文件名列表，例如 ["00000.jpg", "00001.jpg", ...]
# video_dir: 视频帧图像的目录
# show_box(box, ax): 可视化 bounding box 的函数
# show_mask(mask, ax, obj_id): 可视化 mask 的函数

# ========== 可调参数 ==========
step_size = 100  # 处理频率
target_label_id = 1  # 指定处理的 label_id
json_path = "bboxes_result_vid3mask.json"  # 载入的bbox标注JSON路径

def main():
    # 验证外部变量已定义（用户需自行定义）
    try:
        _ = predictor, inference_state, frame_names, video_dir, show_box, show_mask
    except NameError:
        raise RuntimeError("请确保 predictor, inference_state, frame_names, video_dir, show_box, show_mask 已在全局作用域中定义")

    total_frames = len(frame_names)
    print(f"===== 现在以 step_size={step_size} 方式处理帧 (即每隔 {step_size} 帧) =====")

    with open(json_path, "r", encoding="utf-8") as f:
        bboxes_dict = json.load(f)

    for ann_frame_idx in range(0, total_frames, step_size):
        key = f"{ann_frame_idx:05d}.jpg"
        if key not in bboxes_dict:
            continue

        bbox_list = bboxes_dict[key]
        if not bbox_list:
            continue

        filtered_boxes = [b for b in bbox_list if b["label_id"] == target_label_id]
        if not filtered_boxes:
            continue

        first_bbox_info = filtered_boxes[0]
        bbox_arr = np.array(first_bbox_info["bbox"], dtype=np.float32)

        print(f"  处理帧 {key}, label_id={target_label_id}, bbox={bbox_arr.tolist()}")

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=target_label_id,
            box=bbox_arr
        )

        plt.figure(figsize=(9, 6))
        plt.title(f"[step={step_size}] frame {key}, label_id={target_label_id}")
        frame_path = os.path.join(video_dir, key)
        plt.imshow(Image.open(frame_path))

        show_box(bbox_arr, plt.gca())
        mask_np = (out_mask_logits[0] > 0.0).cpu().numpy()
        show_mask(mask_np, plt.gca(), obj_id=out_obj_ids[0])

        plt.show()

if __name__ == "__main__":
    main()
