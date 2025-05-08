import os
import random
import shutil
def split_train_val(image_dir, label_dir, root_output_dir, val_ratio=0.2, random_seed=42):
    """
    Args:
        image_dir (str): 原始图像所在的目录 (domain1/images)。
        label_dir (str): 原始标注所在的目录 (domain1/labels)。
        root_output_dir (str): 输出根目录 (domain1)。
        val_ratio (float): 验证集占总数据的比例。
        random_seed (int): 随机种子，保证可重复性。
    """
    random.seed(random_seed)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    num_val = int(len(image_files) * val_ratio)
    val_image_files = image_files[:num_val]
    train_image_files = image_files[num_val:]
    train_image_out_dir = os.path.join(root_output_dir, "train", "images")
    train_label_out_dir = os.path.join(root_output_dir, "train", "labels")
    os.makedirs(train_image_out_dir, exist_ok=True)
    os.makedirs(train_label_out_dir, exist_ok=True)
    val_image_out_dir = os.path.join(root_output_dir, "val", "images")
    val_label_out_dir = os.path.join(root_output_dir, "val", "labels")
    os.makedirs(val_image_out_dir, exist_ok=True)
    os.makedirs(val_label_out_dir, exist_ok=True)
    for image_file in train_image_files:
        image_src = os.path.join(image_dir, image_file)
        image_dst = os.path.join(train_image_out_dir, image_file)
        shutil.copy(image_src, image_dst)
        basename = os.path.splitext(image_file)[0]
        label_file = basename + ".txt"
        label_src = os.path.join(label_dir, label_file)
        label_dst = os.path.join(train_label_out_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)
    for image_file in val_image_files:
        image_src = os.path.join(image_dir, image_file)
        image_dst = os.path.join(val_image_out_dir, image_file)
        shutil.copy(image_src, image_dst)
        basename = os.path.splitext(image_file)[0]
        label_file = basename + ".txt"
        label_src = os.path.join(label_dir, label_file)
        label_dst = os.path.join(val_label_out_dir, label_file)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)

    print(f"成功复制 {len(train_image_files)} 张图像和标注到 train/images 和 train/labels")
    print(f"成功复制 {len(val_image_files)} 张图像和标注到 val/images 和 val/labels")

if __name__ == "__main__":
    root_dir = "C:/Users/fyq/Desktop/dataset/resized/domain3"
    image_dir = os.path.join(root_dir, "images")
    label_dir = os.path.join(root_dir, "labels")
    validation_ratio = 0.2
    random_seed_val = 42
    split_train_val(image_dir, label_dir, root_dir, validation_ratio, random_seed_val)