from ultralytics import YOLO
if __name__ == '__main__':
    model=YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\train\weights\best.pt")
    source=r"C:\Users\fyq\Desktop\dataset\resized\domain2\images"
    model.predict(source,save=True,imgsz=960)