from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\n12p\weights\best.pt")
    val_results = model.val(data=r"C:\Users\fyq\Desktop\dataset\resized\data3.yaml",
                            batch=16,
                            imgsz=960,
                            plots=True,
                            iou=0.5,
                            device=0,
                            name="val_13p",
                            save=True,
                            )
