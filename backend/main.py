from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
import torch
import io
app=FastAPI()
origins=[
    "http://localhost",
    "http://localhost:8080",
    "*",
    "file://",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
try:
    model_path = r"C:\Users\fyq\Desktop\bs\model\runs\detect\train4\weights\best.pt"
    model_path = r"C:\baidunetdiskdownload\test\test\wight\v11x_KD\weights\best.pt"
    model = YOLO(model_path)
except Exception as e:
    print(f"模型加载失败:{e}")
@app.post("/detect")
async def detect_items(file:UploadFile=File(...)):
    if not file:
        raise HTTPException(status_code=400,detail="没有图像")
    try:
        image_bytes=await file.read()
        image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400,detail="无效图像格式")
    try:
        results=model(image)
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"推理错误:{e}")
    detections = []
    for result in results:
        boxes_data = result.boxes.cpu().numpy()
        if boxes_data is not None and len(boxes_data):
            boxes = boxes_data.xyxy.astype(int).tolist()
            scores = boxes_data.conf.tolist()
            cls_ids = boxes_data.cls.astype(int).tolist()
            names = model.names
            for box, score, cls_id in zip(boxes, scores, cls_ids):
                x1, y1, x2, y2 = box
                confidence = score
                class_name = names[cls_id]
                detections.append({
                    "box": [int(x1),int(y1),int(x2),int(y2)],
                    "label": class_name,
                    "confidence": float(confidence)
                })
    return {"detections": detections}
@app.get("/")
def read_root():
    return {"message":"FastAPI backend is running"}