import torch
from ultralytics import YOLO
student_model =YOLO(r"yolo11n.pt").model
teacher_model = YOLO(r"C:\Users\fyq\Desktop\bs\model\runs\detect\n12\weights\best.pt").model
def print_model_detection_parameters(model, model_name="Model"):
    if model is None:
        print(f"{model_name} is None.")
        return
    try:
        detection_head = model.model[-1]

        if hasattr(detection_head, 'nc') and hasattr(detection_head, 'reg_max'):
            nc = detection_head.nc
            reg_max = detection_head.reg_max
            use_dfl = reg_max > 1  # DFL 通常在 reg_max > 1 时使用
            no = nc + reg_max * 4 if use_dfl else nc + 4  # 如果不用 DFL，通常是 nc + 4 (bbox) + 1 (obj) or nc + 5
            print(f"--- {model_name} Parameters ---")
            print(f"  Number of Classes (nc): {nc}")
            print(f"  Regression Max (reg_max for DFL): {reg_max}")
            print(f"  Calculated 'no' (outputs per anchor location): {no}")
            print(f"  Detection Head Module: {type(detection_head)}")

            if hasattr(model, 'nc'):
                print(f"  Model's direct nc attribute: {model.nc} (should match head's nc)")

        else:
            print(f"Could not find 'nc' or 'reg_max' attributes in the last layer of {model_name}.")
            print(f"Last layer type: {type(detection_head)}")

    except Exception as e:
        print(f"Error accessing parameters for {model_name}: {e}")
        print(f"Make sure {model_name} is a loaded Ultralytics YOLO-style model and its structure is as expected.")


print_model_detection_parameters(student_model, "Student Model")
print("\n")
print_model_detection_parameters(teacher_model, "Teacher Model")