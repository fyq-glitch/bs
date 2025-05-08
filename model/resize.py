import os
from PIL import Image
classes=[
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
class_id={name: i for i,name in enumerate(classes)}
input_dir=r"C:\Users\fyq\Desktop\dataset\origin"
output_dir=r"C:\Users\fyq\Desktop\dataset\resized"
target_size=960
subdirs=os.listdir(input_dir)
print(f"start resizing: {input_dir}")
for subdir in subdirs:
    input_subdir = os.path.join(input_dir, subdir)
    input_images=os.path.join(input_dir,subdir,'image')
    input_txts= os.path.join(input_dir,subdir,'txt')
    output_subdir = os.path.join(output_dir, subdir)
    output_images = os.path.join(output_dir, subdir, 'images')
    output_labels = os.path.join(output_dir, subdir, 'labels')
    if not os.path.exists(input_subdir):
        print(f"{input_subdir} does not exits")
        continue
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    images_file=sorted([f for f in os.listdir(input_images) if os.path.isfile(os.path.join(input_images,f))])
    txts_file=sorted([f for f in os.listdir(input_txts) if os.path.isfile(os.path.join(input_txts,f))])
    image_basenames={os.path.splitext(f)[0] for f in images_file}
    txt_basenames={os.path.splitext(f)[0] for f in txts_file}
    common_basenames=list(image_basenames.intersection(txt_basenames))
    common_basenames.sort()
    if not common_basenames:
        print(f"{input_subdir}中没有匹配的image和txt")
        continue
    processed_count=0
    for base_filename in common_basenames:
        image_filename = None
        txt_filename = None
        for img_f in images_file:
            if os.path.splitext(img_f)[0]==base_filename:
                image_filename=img_f
                break
        for txt_f in txts_file:
            if os.path.splitext(txt_f)[0]==base_filename:
                txt_filename=txt_f
                break
        if image_filename is None or txt_filename is None:
            print(f"没有名为{base_filename}的图片或者txt")
            continue
        input_image_path=os.path.join(input_images,image_filename)
        input_txt_path=os.path.join(input_txts,txt_filename)
        output_image_path=os.path.join(output_images,image_filename)
        output_labels_path=os.path.join(output_labels,txt_filename)
        try:
            with Image.open(input_image_path) as img:
                original_width,original_height=img.size
                scale_x=target_size/original_width
                scale_y=target_size/original_height
                resized_image=img.resize((target_size,target_size))
                resized_image.save(output_image_path)
            yolo_labels=[]
            with open(input_txt_path,'r') as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts=line.split()
                    if len(parts)!=6:
                        print(f"{txt_filename}格式不正确，第{line}行")
                        continue
                    class_name=parts[1]
                    try:
                        original_xmin=float(parts[2])
                        original_ymin=float(parts[3])
                        original_xmax=float(parts[4])
                        original_ymax=float(parts[5])
                    except ValueError:
                        print(f"{txt_filename}中{line}行坐标值无效")
                        continue
                    resized_xmin=original_xmin*scale_x
                    resized_ymin=original_ymin*scale_y
                    resized_xmax=original_xmax*scale_x
                    resized_ymax=original_ymax*scale_y
                    bbox_width=resized_xmax-resized_xmin
                    bbox_height=resized_ymax-resized_ymin
                    center_x=(resized_xmin+resized_xmax)/2
                    center_y=(resized_ymin+resized_ymax)/2
                    yolo_center_x=center_x/target_size
                    yolo_center_y=center_y/target_size
                    yolo_width=bbox_width/target_size
                    yolo_height=bbox_height/target_size
                    if class_name not in class_id:
                        print(f"{txt_filename}中的{class_name}无效")
                        continue
                    id=class_id[class_name]
                    yolo_labels.append(f"{id} {yolo_center_x:.16f} {yolo_center_y:.16f} {yolo_width:.16f} {yolo_height:.16f}")
            with open(output_labels_path,'w') as f:
                for line in yolo_labels:
                    f.write(line+'\n')
            processed_count+=1
            print(f"已处理{image_filename}和{txt_filename}")
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"发生错误:{e}")
    print(f"完成{input_subdir}，成功处理{processed_count}对文件")
print("\n全部完成")


