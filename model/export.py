from ultralytics import YOLO
model=YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\all200\weights\best.pt")
model.export(format="onnx")
