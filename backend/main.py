from fastapi import FastAPI,File,UploadFile,HTTPException,Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import torch
import os
import time
from pathlib import Path
from PIL import ImageDraw,ImageFont
import onnxruntime as ort
from torchvision.ops import nms
import datetime
import base64
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
class_names=[
    'plasticbottle',
    'pressure',
    'lighter',
    'knife',
    'device',
    'powerbank',
    'umbrella',
    'glassbottle',
    'scissor',
    'laptop'
]
allowed_models={
    "yolo11n":"yolo11n.onnx",
    "yolo11s":"yolo11s.onnx",
    "yolo11m":"yolo11m.onnx",
    "yolo11l":"yolo11l.onnx",
    "yolo11x":"yolo11x.onnx",
}
def preprocess(image: Image.Image, size=960):
    img = image.convert("RGB")
    iw, ih = img.size
    scale = min(size / iw, size / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    new_img = Image.new("RGB", (size, size), (114, 114, 114))
    pad_w = (size - nw) // 2
    pad_h = (size - nh) // 2
    new_img.paste(img_resized, (pad_w, pad_h))
    img_array = np.array(new_img, dtype=np.float32) / 255.0
    img_array = img_array.transpose(2, 0, 1)[None, ...]
    return img_array, scale, pad_w, pad_h

def postprocess(outputs: np.ndarray, scale: float, pad_w: int, pad_h: int,
                conf_thres: float = 0.5,
                iou_thres: float = 0.45):
    preds = outputs[0]
    preds = preds.transpose(1, 0)
    xc,yc,w,h=preds[:,0],preds[:,1],preds[:,2],preds[:,3]
    x1=xc-w/2
    y1=yc-h/2
    x2=xc+w/2
    y2=yc+h/2
    boxes = np.stack([x1,y1,x2,y2],axis=1)
    cls_scores = preds[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confidences = cls_scores[np.arange(len(class_ids)), class_ids]
    mask = confidences > conf_thres
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    boxes -= np.array([pad_w, pad_h, pad_w, pad_h])
    boxes /= scale
    boxes = boxes.clip(min=0)
    keep_indices = nms(
        torch.from_numpy(boxes).float(),
        torch.from_numpy(confidences).float(),
        iou_thres
    ).numpy()

    detections = []
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx].astype(int).tolist()
        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": float(confidences[idx]),
            "label": class_names[int(class_ids[idx])]
        })
    return detections

def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        confidence = det["confidence"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1, y1 - 24), f"{label} {confidence:.2f}", fill="red",font = ImageFont.truetype("arial.ttf",size=20))
    return image

@app.post("/detect/")
async def detect_items(file: UploadFile = File(...),confidence_threshold:float=Form(0.5),iou_threshold:float=Form(0.5),model_name:str=Form('yolo11n')):
    if not file:
        raise HTTPException(status_code=400, detail="没有图像文件")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="无效图像格式")
    model_path=os.path.join("C:/Users/fyq/Desktop/bs/backend/models",allowed_models[model_name])
    session = ort.InferenceSession(model_path,providers=["TensorrtExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # tuili
    img_array, scale, pad_w, pad_h = preprocess(img)
    start_time = time.perf_counter()
    ort_outs = session.run([output_name], {input_name: img_array})
    end_time=time.perf_counter()
    detection_time=(end_time-start_time)*1000
    # 保存
    detections = postprocess(ort_outs[0], scale, pad_w, pad_h,conf_thres=confidence_threshold,iou_thres=iou_threshold)
    annotated_img = draw_detections(img.copy(), detections)
    buffered=io.BytesIO()
    annotated_img.save(buffered,format="JPEG")
    img_bytes=buffered.getvalue()
    annotated_img_base64=base64.b64encode(img_bytes).decode('utf-8')
    """
    annotated_img.save(result_path)
    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(detections,f,indent=4)
    """
    return {
        "message": "推理完成，已保存检测结果",
        "annotated_image_base64": annotated_img_base64,
        "detections": detections,
        "detection_time":round(detection_time,2),
    }
@app.get("/")
def read_root():
    return {"message": "FastAPI ONNX backend running"}

