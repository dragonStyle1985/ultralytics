import urllib.parse

from flask import Flask, request, jsonify, Response
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('E:/workspace/python/ultralytics/yolov8n.pt')

# # Define path to video file
# source = 'ultralytics/assets/shanghai.mp4'
#
# # Run inference on the source
# results = model(source, stream=True)  # generator of Results objects
# print(results)


app = Flask(__name__)


@app.route('/find_boxes/<path:media_path>')
def find_boxes(media_path):
    """
    eg.
    http://localhost:5000/find_boxes/E:/workspace/python/ultralytics/ultralytics/assets/bus.jpg
    Args:
        media_path:

    Returns:

    """
    # URL 解码路径
    decoded_path = urllib.parse.unquote(media_path)

    results_list = []

    # 运行模型推断
    results = model(decoded_path)  # 使用从路径参数获得的路径
    for r in results:
        boxes = r.boxes
        results_list.append({
            'class': boxes.cls,
            'confidence': boxes.conf,
            'speed': r.speed
        })

    # 将 PyTorch 张量转换为列表
    for result in results_list:
        if 'class' in result:
            result['class'] = result['class'].tolist()
        if 'confidence' in result:
            result['confidence'] = result['confidence'].tolist()

    return jsonify(results_list)


@app.route('/count_person/<path:media_path>')
def count_person(media_path):
    """
    目前只能数第一帧的。
    eg.
    http://localhost:5000/count_person/E:/workspace/python/ultralytics/ultralytics/assets/image1.jpg
    Args:
        media_path:

    Returns:

    """
    # URL 解码路径
    decoded_path = urllib.parse.unquote(media_path)

    results_list = []

    # 运行模型推断
    results = model(decoded_path)  # 使用从路径参数获得的路径
    for r in results:
        boxes = r.boxes
        results_list.append({
            'class': boxes.cls,
            'confidence': boxes.conf,
            'speed': r.speed
        })

    # 将 PyTorch 张量转换为列表
    for result in results_list:
        if 'class' in result:
            result['class'] = result['class'].tolist()
        if 'confidence' in result:
            result['confidence'] = result['confidence'].tolist()

    # 计算0的数量
    count_zeros = results_list[0]['class'].count(0)

    # 将整数转换为字符串，并返回
    return Response(str(count_zeros), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
