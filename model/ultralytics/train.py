from distillation.DistillationTrainer import DistillationTrainer
if __name__ == '__main__':
    args=dict(
        model=r"yolo11n.pt",
        data=r"C:\Users\fyq\Desktop\dataset\resized\data12.yaml",
        epochs=100,
        batch=16,
        imgsz=960,
        name="n12sd"
    )

    teacher_weights=r"C:\Users\fyq\Desktop\bs\model\runs\detect\n12\weights\best.pt"
    kdcls_weight = 1.0
    kddfl_weight = 1.0
    kdf_weight=1.0
    temperature=4.0
    trainer=DistillationTrainer(overrides=args,teacher_weights=teacher_weights,kdcls_weight=kdcls_weight,kddfl_weight=kddfl_weight,kdf_weight=kdf_weight,temperature=temperature)
    trainer.train()
