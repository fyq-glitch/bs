from ultralytics import YOLO
if __name__ == '__main__':
    #resume=True
    #r"C:\Users\fyq\Desktop\bs\model\runs\detect\train8\weights\last.pt"
    model = YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\all200\weights\last.pt")
    results = model.train(data=r"C:\Users\fyq\Desktop\data\all.yaml",
                          #数据增强
                          cfg=r"C:\Users\fyq\Desktop\dataset\resized\cfg.yaml",
                          epochs=200,
                          imgsz=960,
                          batch=16,
                          device=0,
                          name="all200",
                          resume=True)