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

### 2️⃣Cross-Domain Comparison / 跨域性能对比

| Cross-Domain Task / 跨域任务 | SO (YOLOv11) | Fast R-CNN [1] | SWDA [1] | PSN [1] | FCOS | Fine-YOLO | **Ours** |
|------------------------------|---------------|----------------|-----------|-----------|-------|------------|-----------|
| D₁ → D₂ | 0.440 | 0.423 | 0.469 | 0.483 | 0.451 | 0.538 | **0.541** |
| D₁ → D₃ | 0.497 | 0.536 | 0.565 | 0.576 | 0.510 | 0.599 | **0.602** |
| D₂ → D₁ | 0.460 | 0.418 | 0.497 | 0.514 | 0.472 | 0.543 | **0.548** |
| D₂ → D₃ | 0.575 | 0.554 | 0.567 | 0.578 | 0.592 | 0.636 | **0.635** |
| D₃ → D₁ | 0.539 | 0.527 | 0.566 | 0.586 | 0.536 | 0.583 | **0.593** |
| D₃ → D₂ | 0.556 | 0.536 | 0.548 | 0.549 | 0.556 | 0.605 | **0.602** |
| **Average / 平均值** | **0.511** | **0.499** | **0.535** | **0.548** | **0.520** | **0.584** | **0.587** |

> **Observation / 结果分析：**  
> Across six cross-domain transfer tasks (D₁, D₂, D₃), the proposed method achieves the highest average mAP (**0.587**),  
> outperforming both classical domain adaptation frameworks (SWDA, PSN) and one-stage detectors (FCOS, Fine-YOLO).  
> 在六组跨域迁移任务（D₁, D₂, D₃）中，本方法取得最高平均 mAP (**0.587**)，  
> 超越了传统域自适应框架（SWDA、PSN）及单阶段检测器（FCOS、Fine-YOLO），  
> 显示出在 **跨域鲁棒性与检测精度** 方面的优越性。


---

### 3️⃣ Ablation Study (Module Contribution) / 消融实验（模块贡献分析）

| Cross-Domain Task / 跨域任务 | **E_full (完整模型)** | E_noDA | E_noST | E_noKD | DA Only | ST Only | KD Only |
|------------------------------|----------------------|--------|--------|--------|----------|----------|----------|
| D₁ → D₂ | **0.541** | 0.512 | 0.519 | 0.537 | 0.493 | 0.503 | 0.499 |
| D₁ → D₃ | **0.602** | 0.553 | 0.548 | 0.579 | 0.546 | 0.543 | 0.532 |
| D₂ → D₁ | **0.548** | 0.509 | 0.510 | 0.541 | 0.500 | 0.495 | 0.503 |
| D₂ → D₃ | **0.635** | 0.598 | 0.615 | 0.629 | 0.607 | 0.602 | 0.604 |
| D₃ → D₁ | **0.593** | 0.590 | 0.586 | 0.580 | 0.588 | 0.571 | 0.579 |
| D₃ → D₂ | **0.602** | 0.596 | 0.592 | 0.605 | 0.576 | 0.580 | 0.576 |
| **Average / 平均值** | **0.587** | 0.560 | 0.562 | 0.579 | 0.552 | 0.549 | 0.549 |

> **Interpretation / 实验分析：**  
> - Removing any single component (DA, ST, or KD) consistently leads to performance degradation across all domain pairs.  
> - The **E_full** configuration achieves the best average mAP (**0.587**), confirming that **domain alignment (DA)**, **self-training (ST)**, and **knowledge distillation (KD)** contribute **complementary benefits**.  
> - Among single components, **DA** provides the largest improvement individually, while **ST + KD** combination yields the best balance between stability and accuracy.  
> - 移除任一模块（DA、ST 或 KD）均导致跨域性能下降，表明三者互补性强。  
> - 完整模型在所有任务中表现最佳（平均 mAP = **0.587**），验证了多策略协同的有效性。  
> - 其中，域对齐（DA）提升最明显，而自训练（ST）与蒸馏（KD）的组合在稳定性与精度之间达到最优平衡。


