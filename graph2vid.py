import os
import cv2

def images_to_video(image_folder, output_video, fps=30.0):
    # 获取图像文件列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    if not images:
        print("未找到任何图片文件。")
        return

    # 获取图像尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"无法读取图像: {first_image_path}")
        return

    height, width, layers = frame.shape

    # 设置视频编码器和输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"警告：跳过无法读取的图像 {image_path}")

    video.release()
    print(f"视频已保存为: {output_video}")

if __name__ == "__main__":
    image_folder =# 修改为你的图片路径
    output_video =  # 输出视频文件路径
    images_to_video(image_folder, output_video)
