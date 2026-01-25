import os
from pathlib import Path
from ultralytics import YOLO
source_data_cfg  = r"C:\Users\fyq\Desktop\data\domain3"
target_img_dir   = r"C:\Users\fyq\Desktop\data\domain1\images"
pseudo_label_dir = r"C:\Users\fyq\Desktop\data\pseudo31p\labels"
teacher_weights  = r"C:\Users\fyq\Desktop\bs\model\runs\detect\n31p\weights\best.pt"
def main():
    os.makedirs(pseudo_label_dir, exist_ok=True)
    model = YOLO(teacher_weights)
    print(f"[Info] Running inference on images in: {target_img_dir}")
    results_list = model.predict(source=target_img_dir, imgsz=960, conf=0.5)
    print(f"[Info] Inference complete. Processing results and saving pseudo-labels.")
    for results in results_list:
        img_path = Path(results.path)
        lines = []
        for r in results:
            if len(r.boxes.cls) > 0:
                cls = int(r.boxes.cls[0])
                x_c, y_c, w, h = r.boxes.xywhn[0].tolist()
                lines.append(f"{cls} {x_c} {y_c} {w} {h}\n")
        txt_file = Path(pseudo_label_dir) / f"{img_path.stem}.txt"
        txt_file.write_text(''.join(lines))
    print(f"[Info] Pseudo-labels generation complete. Files saved to {pseudo_label_dir}")
if __name__ == '__main__':
    main()
