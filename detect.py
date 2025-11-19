import os
import json
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载分类模型
classification_model = torch.load(r'D:\detect\model_fold_5.pth')
classification_model.eval()
classification_model.to(device)

# 加载YOLO模型
yolo_model = YOLO(r"D:\yolov11\runs\detect\train7\weights\best.pt")
yolo_model.to(device)

# 图像预处理
transform = transforms.Compose([
    transforms.ToTensor()
])

def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2]

def filter_instrument_changes(previous_instruments, current_instruments, model_names):
    previous_ids = {int(cls_id) for cls_id, _, _ in previous_instruments}
    current_ids = {int(cls_id) for cls_id, _, _ in current_instruments}
    
    added = [{
        "name": model_names.get(class_id, "Unknown"),
        "confidence": conf,
        "bbox": bbox,
        "center": get_bbox_center(bbox)
    } for class_id, conf, bbox in current_instruments if class_id not in previous_ids]
    
    removed = [{
        "name": model_names.get(class_id, "Unknown"),
        "confidence": conf,
        "bbox": bbox,
        "center": get_bbox_center(bbox)
    } for class_id, conf, bbox in previous_instruments if class_id not in current_ids]
    
    return added, removed

def detect_instruments(yolo_model, frame):
    instruments = []
    results = yolo_model(frame)
    for result in results:
        for box in result.boxes:
            instruments.append((box.cls.item(), box.conf.item(), box.xyxy[0].tolist()))
    return instruments

def main():
    video_path = r"D:\detect\2017vid1.mp4"
    cap = cv2.VideoCapture(video_path)

    prev_class = None
    frame_count = 0
    key_frames = []
    previous_instruments = []
    disappeared_instruments = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        input_image = transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = classification_model(input_image)
            _, predicted_class = outputs.max(1)
            current_class = predicted_class.item()
        
        instruments_current = detect_instruments(yolo_model, frame)
        instrument_info = [{
            "name": yolo_model.names.get(cls_id, "Unknown"),
            "confidence": conf,
            "bbox": bbox,
            "center": get_bbox_center(bbox)
        } for cls_id, conf, bbox in instruments_current]

        print(f"[INFO] 帧 {frame_count}: 预测类别 {current_class}")
        print(f"  检测到的器械: {instrument_info}")

        if len(instruments_current) > 0:
            current_class = len(instruments_current)

        if prev_class is not None and current_class != prev_class:
            print(f"[INFO] 变化检测: 帧 {frame_count}，类别从 {prev_class} 变为 {current_class}")
            
            new_instruments, removed_instruments = filter_instrument_changes(
                previous_instruments, instruments_current, yolo_model.names
            )

            if new_instruments or removed_instruments:
                key_frames.append({
                    'frame_index': frame_count,
                    'previous_class': prev_class,
                    'current_class': current_class,
                    'new_instruments': new_instruments,
                    'removed_instruments': removed_instruments
                })
            
            for removed in removed_instruments:
                disappeared_instruments.add(removed['name'])

        previous_instruments = instruments_current
        prev_class = current_class
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n[REPORT] 关键帧信息:")
    for frame_info in key_frames:
        print(f"帧 {frame_info['frame_index']}: 类别从 {frame_info['previous_class']} 变为 {frame_info['current_class']}")
        print(f"  新增器械: {frame_info['new_instruments']}")
        print(f"  移除器械: {frame_info['removed_instruments']}")

    print("\n[REPORT] 消失过的器械:")
    for instrument in disappeared_instruments:
        print(f"  {instrument}")

    print("\n[REPORT] 未消失的器械:")
    for instrument in previous_instruments:
        print(f"  {yolo_model.names.get(instrument[0], 'Unknown')}")

if __name__ == "__main__":
    main()
