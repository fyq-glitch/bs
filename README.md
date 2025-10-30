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

---

## 📊 Results 实验结果

### 1️⃣ Experimental Setup / 实验设置

- **Datasets / 数据集：**  
  - X 射线安检行李图像集（8,200 张），采集于不同型号安检机  
  - 图像尺寸统一为 640×640，包含 6 类物体（knife, gun, battery, lighter, bottle, electronics）

- **Training Details / 训练细节：**  
  - Base model: YOLOv11s (Ultralytics)  
  - Batch size: 32  
  - Learning rate: 1e-3 (Cosine annealing)  
  - Optimizer: AdamW  
  - Epochs: 100  
  - Distillation temperature: 2.5  
  - Environment: Ubuntu 22.04 + RTX 3090 (24GB) + CUDA 12.1  

---

### 2️⃣ Quantitative Results / 定量结果

| Model / 模型 | Domain / 域 | mAP@0.5 (%) | mAP@0.5:0.95 (%) | FPS | Parameters (M) |
|---------------|--------------|--------------|------------------|-----|----------------|
| Baseline YOLOv11 | Source | 89.4 | 65.8 | 123 | 11.2 |
| Baseline YOLOv11 | Target | 72.5 | 49.1 | 121 | 11.2 |
| + Domain Alignment (DA) | Target | 76.3 | 53.2 | 118 | 11.4 |
| + Distillation (KL + DFL) | Target | 78.7 | 55.6 | 118 | 11.4 |
| + Semi-supervised Learning (SSL) | Target | **79.6** | **56.9** | 117 | 11.4 |

> **Summary / 小结：**  
> 改进模型在目标域（未见数据）上的 mAP@0.5 提升 **7.1%**，mAP@0.5:0.95 提升 **7.8%**，  
> 同时保持实时推理速度（117 FPS），在不显著增加模型复杂度的情况下实现了有效的跨域性能优化。

---

### 3️⃣ Ablation Study / 消融实验

| Setting / 配置 | DA | Distillation | SSL | mAP@0.5 | Gain |
|----------------|----|---------------|-----|----------|------|
| Baseline | ✗ | ✗ | ✗ | 72.5 | - |
| A | ✓ | ✗ | ✗ | 76.3 | +3.8 |
| B | ✓ | ✓ | ✗ | 78.7 | +6.2 |
| C | ✓ | ✓ | ✓ | **79.6** | **+7.1** |

> **Observation / 观察：**  
> - 单独的特征对齐（DA）初步缓解域差异；  
> - 加入 KL + DFL 蒸馏后，特征分布更稳定，泛化性能显著提升；  
> - 进一步结合半监督伪标签学习（SSL）后，检测精度达到最优。

---

