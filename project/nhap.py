import time
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import pytesseract  # Cài đặt T
# Khởi tạo model YOLO
model_path = r"E:\pythonProject\pythonProject1\runs\detect\train6\weights\best.pt"
model = YOLO(model_path)

# Sử dụng VideoCapture với độ phân giải thấp hơn
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Đặt chiều rộng khung hình
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Đặt chiều cao khung hình

frame_skip = 2  # Chỉ xử lý mỗi 2 frame để giảm tải
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chỉ xử lý mỗi `frame_skip` frame
    if frame_count % frame_skip == 0:
        start_time = time.time()

        # Phát hiện vùng thông tin CCCD
        results = model(frame)
        result = results[0]  # Lấy kết quả cho frame đầu tiên

        # Truy cập các box phát hiện và nhãn
        boxes = result.boxes
        labels = result.names

        # Vẽ bounding boxes lên ảnh
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
            label = labels[int(box.cls.item())]
            cv2.putText(frame, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Crop và xử lý OCR nếu có vùng CCCD
        cropped_img = CropImg(boxes, frame)
        if cropped_img is not None:
            processed_img = Img_Processing(cropped_img)
            ocr_result = OCR(processed_img)
            print("OCR Result:", ocr_result)

        # Tính toán FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Hiển thị ảnh với bounding boxes và nhãn
    cv2.imshow("Original Frame with Labels", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
