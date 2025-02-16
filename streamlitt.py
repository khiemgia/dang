import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load mô hình YOLO đã train
MODEL_PATH = "yolo11n.pt"
model = YOLO(MODEL_PATH)

def main():
    st.title("💊 Nhận diện & Phân loại Thuốc bằng YOLOv11")
    st.write("Vui lòng upload ảnh chứa viên thuốc để nhận diện.")

    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

        # Chuyển ảnh sang OpenCV
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Chạy mô hình YOLO
        results = model.predict(img_bgr)

        # Hiển thị kết quả nhận diện
        for result in results:
            img_result = result.plot()
            st.image(img_result, caption="Kết quả nhận diện", use_column_width=True)

            st.write("**Danh sách thuốc nhận diện:**")
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                st.write(f"- **{model.names[cls]}** (Độ chính xác: {conf:.2f})")

if __name__ == "__main__":
    main()

