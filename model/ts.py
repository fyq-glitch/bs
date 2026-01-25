import os
import yaml
from pathlib import Path
from ultralytics import YOLO
source_data_cfg  = r"C:\Users\fyq\Desktop\data\domain1"
target_img_dir   = r"C:\Users\fyq\Desktop\data\domain2\images"
pseudo_label_dir = r"C:\Users\fyq\Desktop\data\domain2\plabels"
teacher_weights  = r"C:\Users\fyq\Desktop\bs\model\runs\detect\n12\weights\best.pt"
student_weights  = r"C:\Users\fyq\Desktop\bs\model\runs\detect\n12\weights\best.pt"
def generate_pseudo_labels(model, image_dir, save_dir, conf):
    os.makedirs(save_dir, exist_ok=True)
    for img_path in sorted(Path(image_dir).glob('*.jpg')):
        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        boxes = results[0].boxes.cpu().numpy()
        h, w = results[0].orig_shape
        lines = []
        for *bbox, conf, cls in boxes:
            x1, y1, x2, y2 = bbox
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{int(cls)} {xc:.16f} {yc:.16f} {bw:.16f} {bh:.16f}\n")
        txt_file = Path(save_dir) / f"{img_path.stem}.txt"
        txt_file.write_text(''.join(lines))
    print(f"[Info] Pseudo-labels saved to {save_dir}")
def main():
    teacher = YOLO(teacher_weights)
    print("[Start] Generating pseudo labels...")
    generate_pseudo_labels(teacher, target_img_dir, pseudo_label_dir, conf=0.7)
    print("[Start] Training student model...")
    student = YOLO(student_weights)
    student.train(
        data="",
        epochs=100,
        imgsz=960,
        batch=16,
        name='ts',
    )
if __name__ == '__main__':
    main()