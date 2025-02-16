from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")  # Load pre-trained YOLO model
    data = r'C:\Users\LENOVO\Downloads\Pill detection.v14i.yolov11\data.yaml'

    # **Train YOLOv5**
    model.train(
        data=data,  # Path to the YAML file
        epochs=20,  # Number of epochs
        imgsz=640,  # Image size for resizing
        project="pill-detection-khiem",  # Output project directory
        name="v14-1",  # Experiment name
        device=0  # Set GPU device (use "cpu" nếu không có GPU)
    )
    print("Training hoàn tất!")

