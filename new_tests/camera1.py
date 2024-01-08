import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型
model = YOLO("../yolov8n.pt")

# 获取摄像头内容，参数 0 表示使用默认的摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()  # 读取摄像头的一帧图像

    if success:
        model.predict(source=frame, show=True)  # 对当前帧进行目标检测并显示结果

    # 通过按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭OpenCV窗口
