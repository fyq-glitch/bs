from ultralytics import YOLO

if __name__ == '__main__':
    model=YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\n12\weights\best.pt")
    source=r"C:\Users\fyq\Desktop\data\domain2\images\00002.jpg"
    results=model.predict(source,imgsz=960,conf=0.7,save=True,show=True)
    for r in results:
        print(r.boxes.xywhn)
        print(r.boxes.cls)
        print(r.boxes.id)
