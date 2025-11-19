import os
import json
import cv2
import numpy as np

def main():
    # ========== 1. 定义你的 RGB->label 映射 ==========
    label_colors_rgb = {
        (255, 255, 0): 1,
        (0, 255, 0): 2,
        (0, 255, 1): 3,
        (255, 0, 255): 4,
    }

    # 转成 BGR，以便和 OpenCV 读取到的像素值一致
    label_colors_bgr = {(b, g, r): label_id for (r, g, b), label_id in label_colors_rgb.items()}

    # ========== 2. 目录与输出设置 ==========
    mask_dir = "/home/huaxi/Yuheng/sam2/segment-anything-2/testfiles/testmask/2017vid9mask"
    output_json_path = "bboxes_result_2017vid9mask.json"

    file_list = os.listdir(mask_dir)
    image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results_dict = {}

    # ========== 3. 处理每张掩码图，计算 bounding box ==========
    for img_name in image_files:
        img_path = os.path.join(mask_dir, img_name)
        mask_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if mask_img is None:
            print(f"无法读取图像: {img_path}")
            continue

        height, width = mask_img.shape[:2]
        print(f"处理图像: {img_name}, 尺寸: {width} x {height}")

        bboxes_for_this_image = []

        for bgr_color, label_id in label_colors_bgr.items():
            b, g, r = bgr_color
            match_mask = np.all(mask_img == (b, g, r), axis=-1)
            if not np.any(match_mask):
                continue

            coords = np.argwhere(match_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            bboxes_for_this_image.append({
                "label_id": label_id,
                "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)]
            })

        # ========== 4. 按 label_id 从小到大排序 ==========
        bboxes_for_this_image.sort(key=lambda item: item["label_id"])

        results_dict[img_name] = bboxes_for_this_image
        print("----")

    # ========== 5. 写入 JSON ==========
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

    print(f"\nBounding box 信息已保存至: {output_json_path}")

if __name__ == "__main__":
    main()
