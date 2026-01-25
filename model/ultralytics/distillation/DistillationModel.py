from copy import deepcopy

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import BaseModel,yaml_model_load,parse_model
from distillation.DistillationLoss import DistillationLoss
from ultralytics.nn.modules import (
    OBB,
    Detect,
    Pose,
    Segment,
    YOLOESegment,
)
from ultralytics.utils import  LOGGER
from ultralytics.utils.torch_utils import (
    initialize_weights,
    scale_img,
)


class DistillationModel(BaseModel):
    """蒸馏模型."""

    def __init__(self, cfg="yolo11n.yaml", ch=3, nc=None, verbose=True,
                 teacher_weights=None, kdcls_weight=0.0, kddfl_weight=0.0, kdf_weight=0.0,
                 temperature=20.0):  # Added kdf_weight
        """
        初始化蒸馏模型。

        Args:
            cfg (str | dict): 模型配置文件路径或字典。
            ch (int): 输入通道数。
            nc (int, optional): 类别数。
            verbose (bool): 是否显示模型信息。
            teacher_weights (str): 教师模型权重文件的路径。
            kdcls_weight (float): 分类蒸馏损失权重。
            kddfl_weight (float): DFL蒸馏损失权重。
            kdf_weight (float): 特征蒸馏损失权重。
            temperature (float): 蒸馏温度。
        """
        super().__init__()  # 调用 BaseModel 的 __init__ (它本身可能不接受这些参数，但会初始化一些基础)
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model (学生模型)
        self.yaml["channels"] = ch
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}
        self.inplace = self.yaml.get("inplace", True)
        # self.end2end 属性将在下面Detect模块检查后设置

        # 初始化教师模型
        if teacher_weights:
            LOGGER.info(f"Loading teacher model from {teacher_weights}")
            self.teacher_model = YOLO(teacher_weights).model  # 加载教师模型
            self.teacher_model.eval()  # 设置为评估模式
            for param in self.teacher_model.parameters():
                param.requires_grad = False  # 冻结教师模型参数
        else:
            self.teacher_model = None
            LOGGER.warning("Teacher model weights not provided. Distillation will not be fully functional.")

        # 存储蒸馏超参数
        self.kdcls_weight = kdcls_weight
        self.kddfl_weight = kddfl_weight
        self.kdf_weight = kdf_weight  # 新增特征蒸馏权重
        self.temperature = temperature

        # Build strides and set end2end property
        m = self.model[-1]  # 获取最后一个模块，通常是 Detect()
        if isinstance(m, (Detect, Segment, Pose, OBB, YOLOESegment)):
            s = 256  # 2x min stride for stride calculation
            m.inplace = self.inplace
            self.end2end = getattr(m, "end2end", False)  # 设置模型的end2end属性

            # _forward_for_stride 仅用于初始化时计算真实步长
            # 它必须调用一个保证返回标准模型输出（非元组）的方法
            def _forward_for_stride_calc(x_stride_input):
                # 调用 BaseModel 的 _predict_once 来确保获得标准输出格式
                return super(DistillationModel, self)._predict_once(x_stride_input)

            stride_calc_outputs = _forward_for_stride_calc(torch.zeros(1, ch, s, s))

            # 确保 stride_calc_outputs 是列表，即使只有一个输出头
            if not isinstance(stride_calc_outputs, (list, tuple)):
                stride_calc_outputs = [stride_calc_outputs]
            elif isinstance(stride_calc_outputs, tuple) and len(stride_calc_outputs) == 2 and isinstance(
                    stride_calc_outputs[0], (list, tuple)):
                # Handle cases where _predict_once might return (predictions, None) or similar for compatibility
                stride_calc_outputs = stride_calc_outputs[0]
                if not isinstance(stride_calc_outputs, list):
                    stride_calc_outputs = [stride_calc_outputs]

            m.stride = torch.tensor([s / out.shape[-2] for out in stride_calc_outputs if hasattr(out, 'shape')])
            self.stride = m.stride
            if hasattr(m, 'bias_init') and callable(m.bias_init):
                m.bias_init()
        else:
            self.stride = torch.Tensor([32])  # Default stride
            self.end2end = False  # 如果最后一个模块不是Detect类型，则end2end为False

        initialize_weights(self)  # 初始化学生模型的权重
        if verbose:
            self.info()
            LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False, embed=None, return_features_for_distill=False):
        """
        DistillationModel 的前向传播方法。

        Args:
            x (torch.Tensor): 输入张量。
            augment (bool): 是否进行数据增强推理。
            profile (bool): 是否打印每层计算时间。
            visualize (bool): 是否保存特征图。
            embed (list, optional): 需要返回嵌入特征的层索引列表。
            return_features_for_distill (bool): 如果为 True，则为知识蒸馏返回额外的颈部特征图。

        Returns:
            (torch.Tensor) or (tuple(torch.Tensor, list[torch.Tensor])):
                - 如果 return_features_for_distill 为 False，则返回模型的最终输出。
                - 如果 return_features_for_distill 为 True，则返回一个元组 (最终输出, 颈部特征图列表)。
        """
        if self.training and return_features_for_distill:
            y = []
            neck_features_captured = None
            current_tensor_val = x

            for i, m in enumerate(self.model):
                if m.f != -1:
                    input_to_m = y[m.f] if isinstance(m.f, int) else [y[j] for j in m.f]
                else:
                    input_to_m = current_tensor_val

                # if profile: self._profile_one_layer(m, input_to_m, dt) # 假设有_profile_one_layer

                if hasattr(m, 'type') and m.type == 'Detect':
                    # input_to_m 是送入 Detect 模块的颈部特征图列表 [P3, P4, P5]
                    neck_features_captured = input_to_m

                current_tensor_val = m(input_to_m)
                y.append(current_tensor_val if m.i in self.save else None)

                # if visualize: feature_visualization(current_tensor_val, m.type, m.i, save_dir=visualize)
                if embed and m.i in embed:
                    LOGGER.warning("`embed` 功能在蒸馏模式下可能导致意外行为。")
                    # ... (原始 embed 逻辑) ...
                    pass

            final_head_output = current_tensor_val

            if neck_features_captured is None:
                LOGGER.error("未能捕获到用于蒸馏的颈部特征图。请检查模型结构和 'Detect' 层类型。")

            # final_head_output 可能是单个张量（如RTDETR）或列表（如YOLOv8的3个检测头输出）
            # 您的 DistillationLoss 需要能处理这两种情况
            return final_head_output, neck_features_captured

        elif augment:  # 测试时增强
            return self._predict_augment(x)
        else:  # 标准推理或不进行特征蒸馏的训练
            return super()._predict_once(x, profile=profile, visualize=visualize, embed=embed)

    def _predict_augment(self, x):
        """
        对输入图像x执行增强，并返回增强后的推理结果。
        (基本保持您提供的版本)
        """
        if self.end2end or self.__class__.__name__ != "DistillationModel":  # 使用 self.end2end
            LOGGER.warning(
                f"模型 {self.__class__.__name__} (end2end: {self.end2end}) 不支持 'augment=True' 或配置不符。"
                f"将回退到单尺度预测。"
            )
            return super()._predict_once(x), None  # 确保返回元组

        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (None, L-R)

        y_augmented_preds = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            # 调用父类的 predict, 它会调用 _predict_once, 只返回一个张量或元组的第一个元素
            yi_pred_output = super().predict(xi)  # BaseModel.predict -> BaseModel._predict_once

            # 从父类预测中提取主要预测（通常是元组的第一个元素或直接是张量）
            if isinstance(yi_pred_output, (list, tuple)):
                yi_main_pred = yi_pred_output[0]
            else:
                yi_main_pred = yi_pred_output

            yi_processed = self._descale_pred(yi_main_pred, fi, si, img_size)
            y_augmented_preds.append(yi_processed)

        y_clipped = self._clip_augmented(y_augmented_preds)
        return torch.cat(y_clipped, 1) if isinstance(y_clipped, list) and len(y_clipped) > 0 else y_clipped, None

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        (保持您提供的版本)
        """
        # p: [bs, num_dets, nc+5] or [bs, num_queries, nc+5]
        # Ultralytics的 bbox 格式通常是 cxcywh
        # p[:, :4] /= scale # 这种直接除法可能不适用于所有坐标格式

        # 假设 p 的前4个通道是 cx, cy, w, h (归一化到图像尺寸)
        bboxes, scores_classes = p.split([4, p.shape[dim] - 4], dim=dim)

        # 反归一化坐标到绝对像素值 (如果它们是归一化的)
        # 如果已经是绝对像素值，则直接除以scale
        # 假设它们是相对于当前 xi 图像尺寸的归一化 cxcywh
        # 这里需要知道 xi 的尺寸，但我们只有原始 img_size 和 scale
        # 通常的做法是先将预测框缩放到原始图像尺度，然后再处理翻转

        # 为了简化，我们假设这里的 p[:, :4] 是相对于 xi 尺寸的绝对坐标
        # 并且可以直接通过 scale 映射回原始 x 的尺度空间
        bboxes_scaled = bboxes / scale  # De-scale bounding box coordinates

        # 分离坐标
        cx, cy, w, h = bboxes_scaled.split((1, 1, 1, 1),
                                           dim=0 if bboxes_scaled.ndim == 1 else 1)  # dim 0 if 1D, 1 if 2D etc.
        # More robust: bboxes_scaled.shape.index(4)-1

        # 处理翻转 (在原始图像坐标系下)
        if flips == 2:  # up-down
            cy = img_size[0] - cy
        elif flips == 3:  # left-right
            cx = img_size[1] - cx

        return torch.cat((cx, cy, w, h, scores_classes), dim=dim)

    def _clip_augmented(self, y):
        """
        (保持您提供的版本，但要注意其对Detect模块内部结构的依赖)
        """
        if not y or not all(isinstance(i, torch.Tensor) for i in y):  # y is a list of tensors
            LOGGER.warning("_clip_augmented received empty or invalid input.")
            return y if y else []  # Return empty list if y is empty

        # 尝试获取 nl (number of detection layers)
        # 这部分比较依赖 Detect 模块的内部实现，如果 Detect 模块没有 nl 属性，会出错
        try:
            nl = self.model[-1].nl
        except AttributeError:
            LOGGER.warning("Detect module does not have 'nl' attribute. Clipping might be incorrect or skipped.")
            # 如果没有nl，无法执行原裁剪逻辑，可以选择不裁剪或用默认值
            return y  # 返回未裁剪的

        g = sum(4 ** x for x in range(nl))
        e = 1  # exclude layer count

        # 大尺度预测的裁剪 (通常是列表中的第一个元素 y[0])
        if y[0].shape[-1] > 0:  # 确保张量不为空
            i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e)) if g > 0 else 0
            if i < y[0].shape[-1]:  # 确保索引不越界
                y[0] = y[0][..., :-i] if i > 0 else y[0]
            else:  # 如果i过大，说明有问题，可能不裁剪或记录错误
                LOGGER.warning("Clipping index for large scale predictions is out of bounds.")

        # 小尺度预测的裁剪 (通常是列表中的最后一个元素 y[-1])
        if len(y) > 1 and y[-1].shape[-1] > 0:  # 确保有小尺度预测且不为空
            i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e)) if g > 0 else 0
            if i < y[-1].shape[-1]:  # 确保索引不越界
                y[-1] = y[-1][..., i:] if i > 0 else y[-1]
            else:  # 如果i过大
                LOGGER.warning("Clipping index for small scale predictions is out of bounds.")
        return y

    def init_criterion(self):
        """初始化模型的损失标准。"""
        # 确保教师模型已加载
        if not self.teacher_model:
            LOGGER.error("Teacher model is not loaded. Cannot initialize distillation criterion.")
            return None  # 或者抛出异常

        # 这里的 self 就是 DistillationModel 实例 (学生模型)
        return DistillationLoss(model=self, teacher_model=self.teacher_model,
                                kdcls_weight=self.kdcls_weight, kddfl_weight=self.kddfl_weight,
                                kdf_weight=self.kdf_weight,  # 确保传递 kdf_weight
                                temperature=self.temperature)
