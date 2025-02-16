from ultralytics import YOLO
import os
import shutil
model = YOLO("yolo11n.pt")  # Load pre-trained YOLOv5 model
data = r'C:\Users\LENOVO\Downloads\Pill detection.v14i.yolov11\data.yaml'
# **4. Train YOLOv5**
# Fine-tuning the YOLO model using your dataset
model.train(
    data=data,  # Path to the YAML file
    epochs=20,  # Number of epochs
    imgsz=640,  # Image size for resizing
    project="pill-detection-khiem",  # Output project directory
    name="v14-1",  # Experiment name
    device=0  # Set GPU device (use "cpu" if no GPU is available)
)

test_image_path = r"C:\Users\LENOVO\Downloads\Pill detection.v14i.yolov11\test\images"
save_directory = r"C:\Users\LENOVO\Downloads\Pill detection.v14i.yolov11\runs\detect\predict"

predictions = model.predict(source=test_image_path, save=True, show=True, project = "pill-detection-khiem")

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load mô hình YOLO đã train
MODEL_PATH = "yolo11n.pt"  # Thay thế bằng đường dẫn mô hình của bạn
model = YOLO(MODEL_PATH)

# Streamlit UI
st.title("💊 Nhận diện & Phân loại Thuốc bằng YOLOv11")
st.write("Vui lòng upload ảnh chứa viên thuốc để nhận diện.")

# Upload ảnh
uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Chuyển ảnh sang định dạng OpenCV
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Chạy mô hình YOLO để nhận diện
    results = model.predict(img_bgr)

    # Hiển thị kết quả nhận diện
    for result in results:
        img_result = result.plot()  # Vẽ khung nhận diện lên ảnh
        st.image(img_result, caption="Kết quả nhận diện", use_column_width=True)

        # Hiển thị thông tin thuốc được nhận diện
        st.write("**Danh sách thuốc nhận diện:**")
        for box in result.boxes:
            cls = int(box.cls[0])  # Lớp nhận diện
            conf = box.conf[0]  # Độ chính xác
            st.write(f"- **{model.names[cls]}** (Độ chính xác: {conf:.2f})")
