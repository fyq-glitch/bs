from distillation.DistillationTrainer import DistillationTrainer
if __name__ == '__main__':
    args=dict(
        model=r"C:\Users\fyq\Desktop\ultralytics\cfg\models\11\yolo11n.yaml",
        data=r"C:\baidunetdiskdownload\test\test\KD_test\CLC_extract\data.yaml",
        epochs=100,
        batch=8,
        imgsz=960,
    )
    #蒸馏参数
    teacher_weights=r"C:\baidunetdiskdownload\test\test\wight\v11x_KD\weights\test.pt"
    kdcls_weight = 1.0
    kddfl_weight = 1.0
    temperature=4.0
    trainer=DistillationTrainer(overrides=args,teacher_weights=teacher_weights,kdcls_weight=kdcls_weight,kddfl_weight=kddfl_weight,temperature=temperature)
    trainer.train()
