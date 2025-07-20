import os
import cv2
import shutil
import random

# Đường dẫn đến thư mục chứa ảnh và nhãn
images_dir = 'E:/pythonProject/pythonProject1/cccd_image3/images'
labels_dir = 'E:/pythonProject/pythonProject1/cccd_image3/labels'

# Tạo thư mục cho tập train và val
os.makedirs('images/train', exist_ok=True)
os.makedirs('images/val', exist_ok=True)
os.makedirs('labels/train', exist_ok=True)
os.makedirs('labels/val', exist_ok=True)

# Lấy danh sách tất cả các tệp ảnhLa
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

# Đảo ngẫu nhiên danh sách ảnh
random.shuffle(image_files)

# Tính toán số lượng ảnh cho tập huấn luyện và kiểm tra
train_size = int(len(image_files) * 0.8)  # 80% cho tập huấn luyện
train_files = image_files[:train_size]
val_files = image_files[train_size:]

# Di chuyển các tệp ảnh và nhãn vào thư mục tương ứng
for image_file in train_files:
    # Di chuyển ảnh vào thư mục train
    shutil.move(os.path.join(images_dir, image_file), os.path.join('images/train', image_file))

    # Di chuyển tệp nhãn tương ứng vào thư mục train
    label_file = image_file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(labels_dir, label_file)):
        shutil.move(os.path.join(labels_dir, label_file), os.path.join('labels/train', label_file))
    else:
        print(f"Warning: Label file for {image_file} not found.")

for image_file in val_files:
    # Di chuyển ảnh vào thư mục val
    shutil.move(os.path.join(images_dir, image_file), os.path.join('images/val', image_file))

    # Di chuyển tệp nhãn tương ứng vào thư mục val
    label_file = image_file.replace('.jpg', '.txt')
    if os.path.exists(os.path.join(labels_dir, label_file)):
        shutil.move(os.path.join(labels_dir, label_file), os.path.join('labels/val', label_file))
    else:
        print(f"Warning: Label file for {image_file} not found.")
