# Cross-Domain Object Detection Optimization Based on YOLOv11  
基于 YOLOv11 的跨域目标检测优化研究

---

## 📘 Overview 概述

This repository implements a **domain-adaptive YOLOv11** framework designed to improve object detection performance under **cross-domain shifts** in **X-ray security imagery**.  
本项目实现了一个 **面向 X 射线安检图像的 YOLOv11 域自适应检测框架**，通过 **知识蒸馏、特征空间对齐** 与 **半监督学习**，提升模型在跨域场景下的检测性能。

---

## 🚀 Highlights 研究亮点

- **Domain Adaptation / 域自适应：** 针对安检图像领域间分布差异导致的性能下降问题。  
- **Teacher–Student Distillation / 师生蒸馏：** 结合 KL 散度与 DFL 损失，实现跨域特征迁移。  
- **Feature Alignment / 特征对齐：** 在特征空间中减少源域与目标域分布偏移。  
- **Semi-Supervised Learning / 半监督学习：** 利用伪标签和一致性约束提升泛化能力。  
- **Performance Boost / 性能提升：** 在未见目标域上实现 **+7.1% mAP 增益**，显著提升跨域鲁棒性。

---

## 🧠 Motivation 研究动机

In X-ray security applications, models trained on one dataset often fail to generalize to unseen environments due to **domain shifts** (e.g., device type, background, or lighting).  
在 X 射线安检场景中，不同设备或数据集之间存在显著的 **域差异（domain shift）**，导致模型泛化性能下降。  
本研究尝试利用 **蒸馏学习 + 特征对齐策略** 缓解这一问题，从而实现高鲁棒性的跨域检测。

---

## ⚙️ Implementation 实现细节

- **Base Model:** YOLOv11 (Ultralytics)  
- **Frameworks:** PyTorch, OpenCV, NumPy  
- **Loss Function:** KL Divergence + Distribution Focal Loss (DFL)  
- **Adaptation Strategy:** Semi-supervised pseudo-labeling & feature alignment  
- **Visualization:** t-SNE embedding and OpenCV domain comparison  

---

## 📊 Results 实验结果

2️⃣ Quantitative Results / 定量结果
Model / 模型	Domain / 域	mAP@0.5 (%)	mAP@0.5:0.95 (%)	FPS	Parameters (M)
Baseline YOLOv11	Source	89.4	65.8	123	11.2
Baseline YOLOv11	Target	72.5	49.1	121	11.2
+ Domain Alignment (DA)	Target	76.3	53.2	118	11.4
+ Distillation (KL + DFL)	Target	78.7	55.6	118	11.4
+ Semi-supervised Learning (SSL)	Target	79.6	56.9	117	11.4

Summary / 小结：

在目标域（未见数据）上，改进模型的 mAP@0.5 提升了 7.1%，mAP@0.5:0.95 提升了 7.8%，
同时保持实时推理速度（117 FPS），证明本方法在不显著增加模型复杂度的情况下实现了有效的跨域优化。

3️⃣ Ablation Study / 消融实验
Setting / 配置	DA	Distillation	SSL	mAP@0.5	Gain
Baseline	✗	✗	✗	72.5	-
A	✓	✗	✗	76.3	+3.8
B	✓	✓	✗	78.7	+6.2
C	✓	✓	✓	79.6	+7.1

Observation / 观察：

单独的特征对齐（DA）能初步缓解域差异。

引入 KL + DFL 蒸馏后，模型特征分布更稳定，显著提升泛化性能。

在此基础上加入半监督学习（SSL）后，伪标签机制进一步提升目标域精度。

---

