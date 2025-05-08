import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    freeze_support()

    # 加载自定义权重
    model_path = r"C:\baidunetdiskdownload\test\test\wight\v11x_KD\weights\best.pt"
    model = YOLO(model_path)

    images_folder = r"C:\baidunetdiskdownload\test\test\KD_test\CLC_extract\colorized\images"
    labels_folder = r"C:\baidunetdiskdownload\test\test\KD_test\CLC_extract\colorized\labels"

    '''
    # 推理
    output_folder = "predicted_images"
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
    all_predictions = []
    all_ground_truth = []
    names = model.names
    for image_file in tqdm(image_files, desc="Processing"):
        image_path = os.path.join(images_folder, image_file)
        image = cv2.imread(image_path)
        original_image = image.copy()
        h, w, _ = original_image.shape
        label_file_base = os.path.splitext(image_file)[0]
        label_path = os.path.join(labels_folder, label_file_base + ".txt")
        ground_truth_boxes = []
        ground_truth_classes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * w
                        y_center = float(parts[2]) * h
                        box_width = float(parts[3]) * w
                        box_height = float(parts[4]) * h
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center - box_height / 2)
                        ground_truth_boxes.append([x1, y1, x2, y2])
                        ground_truth_classes.append(class_id)
                    else:
                        print(f"警告: 标签文件 {label_path} 中的行格式不正确: {line}")
        try:
            results = model(image)
            for result in results:
                boxes_data = result.boxes.cpu().numpy()
                if boxes_data is not None and len(boxes_data):
                    boxes = boxes_data.xyxy.astype(int)
                    scores = boxes_data.conf
                    cls_ids = boxes_data.cls.astype(int)
                    for box, score, cls_id in zip(boxes, scores, cls_ids):
                        x1, y1, x2, y2 = box
                        confidence = score
                        class_name = names[cls_id]
                        all_predictions.append({
                            'image_id': image_file,
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'category_id': cls_id
                        })

                        label = f"{class_name} {confidence:.2f}"
                        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(original_image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        except Exception as e:
            print(f"处理图片 {image_file} 时发生推理错误: {e}")
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, original_image)
        if ground_truth_boxes:
            for box, cls_id in zip(ground_truth_boxes, ground_truth_classes):
                all_ground_truth.append({
                    'image_id': image_file,
                    'bbox': box,
                    'category_id': cls_id
                })

    print(f"已预测的图片保存到: {output_folder}")
    '''

    print("\n开始评估模型...")
    try:
        metrics = model.val(data=r"C:/baidunetdiskdownload/test/test/KD_test/CLC_extract/data.yaml")
        print("\n评估结果:")
        print(metrics)


        if hasattr(metrics, 'results_dict'):
            metric_names = ['mAP50', 'mAP50-95', 'Precision', 'Recall', 'Fitness']
            metric_values = [
                metrics.results_dict['metrics/mAP50(B)'],
                metrics.results_dict['metrics/mAP50-95(B)'],
                metrics.results_dict['metrics/precision(B)'],
                metrics.results_dict['metrics/recall(B)'],
                metrics.fitness
            ]

            plt.figure(figsize=(10, 6))
            plt.bar(metric_names, metric_values, color='skyblue')
            plt.ylabel("Value")
            plt.title("Overall Evaluation Metrics")
            plt.ylim(0, 1)
            plt.show()
        else:
            print("无法获取整体评估指标进行可视化。")


        if hasattr(metrics, 'maps') and hasattr(metrics, 'names'):
            ap_per_class = metrics.maps * 100
            class_names = list(metrics.names.values())

            plt.figure(figsize=(14, 7))
            sns.barplot(x=class_names, y=ap_per_class, palette="viridis")
            plt.xlabel("Class Name")
            plt.ylabel("Average Precision (%)")
            plt.title("Average Precision per Class")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            print("无法获取每个类别的平均精度数据进行可视化。")

        print(f"\n混淆矩阵保存在 {metrics.save_dir}/confusion_matrix.png，请查看该路径。")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")