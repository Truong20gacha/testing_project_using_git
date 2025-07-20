import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import cv2
from PIL import Image, ImageTk, ImageOps
import pytesseract
import numpy as np
import os
import csv

from ultralytics.models import YOLO

# Đường dẫn tới mô hình YOLO và Tesseract
model_path = r"E:\pythonProject\pythonProject1\runs\detect\train6\weights\best.pt"
model = YOLO(model_path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Từ điển biệt danh (alias) cho nhãn
LABEL_ALIASES = {
    "name": "Số CMND/CCCD",
    "dob": "Họ và tên",
    "id": "Ngày sinh",
    "quequan": "Quê quán",
    "datnuoc": "Quốc tịch"
}

# Hàm tiền xử lý ảnh
def PreprocessRegion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)
    return inverted

# Hàm OCR với Tesseract
def OCR_Tesseract(img):
    pil_img = Image.fromarray(img)
    config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(pil_img, lang='vie', config=config).strip()

# Hàm nhận diện thông tin từ khung hình
def DetectFromFrame(frame):
    results = model(frame)
    boxes = results[0].boxes
    labels = results[0].names
    data = {}

    for idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        cropped = frame[y_min:y_max, x_min:x_max]
        processed = PreprocessRegion(cropped)
        text_tesseract = OCR_Tesseract(processed)
        label = labels[int(box.cls.item())]
        data[label] = {"Tesseract": text_tesseract}

    return data

# Hàm nhận diện từ file
def DetectFromFile(file_path):
    frame = cv2.imread(file_path)
    if frame is None:
        messagebox.showerror("Lỗi", "Không thể đọc file ảnh.")
        return None

    results = DetectFromFrame(frame)
    DisplayResults(results)

# Nhận diện từ camera
def DetectFromCamera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không nhận được khung hình từ camera.")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Chụp ảnh
            cap.release()
            cv2.destroyAllWindows()
            results = DetectFromFrame(frame)
            DisplayResults(results)
            break
        elif key == ord('q'):  # Thoát
            cap.release()
            cv2.destroyAllWindows()
            break

# Hàm lưu kết quả vào file CSV
def save_to_txt(results, file_path="E:\\pythonProject\\pythonProject1\\project_cccd\\cccd_results.txt"):
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write("=== Kết quả nhận diện ===\n")
            for label, texts in results.items():
                alias = LABEL_ALIASES.get(label, label.capitalize())
                f.write(f"{alias}: {texts['Tesseract']}\n")
                f.write("-" * 30 + "\n")
            f.write("\n")  # Thêm dòng trống giữa các lần lưu
        messagebox.showinfo("Lưu thành công", f"Kết quả đã được lưu vào file TXT: {file_path}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể lưu kết quả vào TXT: {str(e)}")

# Hiển thị kết quả lên giao diện
def DisplayResults(results):
    if results:
        result_text = "=== Kết quả nhận diện ===\n\n"
        for label, texts in results.items():
            alias = LABEL_ALIASES.get(label, label.capitalize())
            result_text += f"{alias}: {texts['Tesseract']}\n"
            result_text += "-" * 30 + "\n"
        result_var.set(result_text)
        save_to_txt(results)  # Lưu vào TXT
    else:
        result_var.set("Không phát hiện được thông tin nào.")


# Hàm tải ảnh và thực hiện nhận diện
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((500, 300), Image.LANCZOS)  # Hiển thị ảnh lớn hơn
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        image_label.image = img_display
        DetectFromFile(file_path)

# Giao diện chính Tkinter
root = tk.Tk()
root.title("Nhận diện thông tin CCCD")
root.geometry("800x400")

# Khung bên trái
frame_left = tk.Frame(root)
frame_left.grid(row=0, column=0, padx=10, pady=20, sticky="n")

# Khung chứa ảnh
frame_image = tk.LabelFrame(frame_left, text="Hình ảnh", padx=10, pady=10)
frame_image.pack()
image_label = tk.Label(frame_image, text="Chọn ảnh để nhận dạng", bg="gray", width=50, height=15)
image_label.pack()

# Các nút chức năng
button_frame = tk.Frame(frame_left)
button_frame.pack(pady=5)
btn_select = tk.Button(button_frame, text="Chọn tệp", command=load_image, width=15, height=2, font=("Arial", 12))
btn_select.grid(row=0, column=0, padx=5)
btn_camera = tk.Button(button_frame, text="Mở camera", command=DetectFromCamera, width=15, height=2, font=("Arial", 12))
btn_camera.grid(row=0, column=1, padx=5)

# Lưu ý
label_note = tk.Label(frame_left, text="Lưu ý:\n- Ảnh là mặt trước CCCD, đủ 4 góc\n- Ảnh rõ thì độ chính xác cao")
label_note.pack()

# Khung bên phải
frame_right = tk.Frame(root)
frame_right.grid(row=0, column=1, padx=10, pady=10, sticky="n")

# Khung kết quả
frame_result = tk.LabelFrame(frame_right, text="Kết quả nhận dạng", padx=20, pady=20)
frame_result.pack(anchor="w", fill="both", expand=True)

# Nhãn hiển thị kết quả
result_var = tk.StringVar()
result_var.set("Kết quả nhận dạng sẽ hiển thị tại đây")
label_result = tk.Label(frame_result, textvariable=result_var, justify="left", anchor="nw", font=("Arial", 12), padx=10, pady=10)
label_result.pack(anchor="nw", fill="both", expand=True)

# Chạy ứng dụng
root.mainloop()
