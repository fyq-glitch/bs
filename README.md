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

| Model / 模型 | Target Domain / 目标域 | mAP (%) | Gain / 提升 |
|---------------|----------------------|----------|--------------|
| Baseline YOLOv11 | Unseen Dataset | 72.5 | - |
| Proposed Method | Unseen Dataset | **79.6** | +7.1 |

<p align="center">
  <img src="images/domain_alignment.png" width="70%">
  <br>
  <em>Feature alignment visualization between source and target domains. / 源域与目标域特征对齐可视化</em>
</p>

---

## 📁 Repository Structure 仓库结构

