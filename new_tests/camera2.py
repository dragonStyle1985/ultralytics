import cv2
from ultralytics import YOLO

# 载入 YOLOv8 模型
model = YOLO('../yolov8n.pt')

# 获取摄像头内容
cap = cv2.VideoCapture(0)

# 获取原视频的宽度和高度
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置新的视频帧大小
new_width = 1280
new_height = 720

# 设置保存视频的文件名、编解码器和帧速率
output_path = "output_video.avi"  # 替换为你的输出视频文件路径
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (original_width, original_height))

# 循环遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 对帧运行 YOLOv8 推理
        results = model(frame)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        # 显示带有标注的帧
        cv2.imshow("YOLOv8 推理", annotated_frame)

        # 将帧写入输出视频
        out.write(frame)

        # 如果按下 'q' 键，则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果达到视频结尾，中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()
out.release()
