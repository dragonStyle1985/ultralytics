import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("../yolov8n.pt")

# 获取摄像头内容
cap = cv2.VideoCapture(0)

# 获取视频内容
# video_path = "1.mp4"  # 替换为你的视频文件路径
# cap = cv2.VideoCapture(video_path)

# 获取原视频的大小
# original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 设置新的视频帧大小
# new_width = 1280
# new_height = 720

# 设置保存视频的文件名、编解码器和帧速率
# output_path = "output_video.avi"  # 替换为你的输出视频文件路径
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(output_path, fourcc, 20.0, (original_width, original_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 调整帧的大小
    # frame = cv2.resize(frame, (new_width, new_height))

    # 使用模型进行目标检测,并返回相应数据
    results_list = model.predict(source=frame)

    # 获取每个结果对象并进行处理
    for results in results_list:
        if results.boxes is not None:
            xyxy_boxes = results.boxes.xyxy
            conf_scores = results.boxes.conf
            cls_ids = results.boxes.cls

            for box, conf, cls_id in zip(xyxy_boxes, conf_scores, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cls_id = int(cls_id)
                label = model.names[cls_id]
                confidence = f"{conf:.2f}"

                # 颜色
                rectangle_color = (0, 255, 0)
                label_color = (0, 0, 255)

                # 在图像上绘制矩形框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), rectangle_color, 2)
                cv2.putText(frame, f"{label} {confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color,2)

    # 显示图像
    cv2.imshow('YOLOv8 Real-time Detection', frame)

    # 将帧写入输出视频
    # out.write(frame)

    # 如果按下 'q' 键，则中断循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频文件和输出视频
cap.release()
cv2.destroyAllWindows()
# out.release()

