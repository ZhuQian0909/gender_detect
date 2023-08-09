import cv2
import sys
import numpy as np

def detect_gender(image_path):
    # 加载预训练的模型
    model = cv2.dnn.readNetFromCaffe(
        "gender_deploy.prototxt",
        "gender_net.caffemodel"
    )

    # 图片预处理
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (227, 227)), 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # 输入图片到网络进行预测
    model.setInput(blob)
    predictions = model.forward()
    gender_index = np.argmax(predictions[0])
    print(predictions)

    if gender_index == 0:
        return "Male"
    else:
        return "Female"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gender_recognition.py path/to/image")
        sys.exit(1)

    image_path = sys.argv[1]
    result = detect_gender(image_path)
    print(f"Detected gender: {result}")
