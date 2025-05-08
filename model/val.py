from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\x12\weights\best.pt")
    val_results = model.val(data=r"C:\Users\fyq\Desktop\dataset\resized\data2.yaml",
                            batch=16,
                            imgsz=960,
                            plots=True,
                            iou=0.5,
                            device=0,
                            name="val_12x",
                            save=True,
                            )
