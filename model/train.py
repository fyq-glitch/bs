from ultralytics import YOLO
if __name__ == '__main__':
    #resume=True
    #r"C:\Users\fyq\Desktop\bs\model\runs\detect\train8\weights\last.pt"
    model = YOLO(r"yolo11n.pt")
    results = model.train(data=r"C:\Users\fyq\Desktop\dataset\resized\data31.yaml",
                          cfg=r"C:\Users\fyq\Desktop\dataset\resized\cfg.yaml",
                          epochs=100,
                          imgsz=960,
                          batch=16,
                          device=0,
                          name="n31p",
                          resume=True)