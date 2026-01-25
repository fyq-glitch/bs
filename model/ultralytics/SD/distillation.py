import random
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel
import math
import time
import warnings
from copy import copy
import numpy as np
import torch
from torch import nn
import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
)
class DistillationTrainer(BaseTrainer):
    """
    A DetectionTrainer that adds knowledge distillation:
      - student: trained as usual by DetectionTrainer
      - teacher: frozen model loaded from weights
      - loss = α * det_loss + (1-α) * KD_loss
    """
    def __init__(
        self,
        overrides=None,
        _callbacks=None
    ):
        overrides = overrides.copy() if overrides is not None else {}
        self.teacher_weights=overrides.pop('teacher_weights', None)
        self.alpha=overrides.pop('alpha', 0.5)
        self.T=overrides.pop('T', 4.0)
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss', 'kd_loss')
    def setup_model(self):
        ckpt=super().setup_model()
        self.student_yolo=de_parallel(self.model).model
        self.teacher = DetectionModel(
            cfg=self.args.model,
            nc=self.data['nc'],
            ch=self.data['channels'],
            verbose=False
        ).to(self.device)
        if self.teacher_weights:
            ck = torch.load(self.teacher_weights, map_location=self.device)
            state = ck.get('model', ck)
            self.teacher.model.load_state_dict(
                state.state_dict() if hasattr(state, 'state_dict') else state,
                strict=False
            )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_yolo=de_parallel(self.teacher).model
        self.teacher.args = self.args
        self.teacher.nc = self.data['nc']
        self.teacher.names = self.data['names']
        self.teacher.stride = de_parallel(self.teacher).stride
        return ckpt

    def _do_train(self, world_size=1):
        """Train with KD incorporated into the standard loop."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Starting KD training: α={self.alpha}, T={self.T}\n"
            f"Logging to {colorstr('bold', self.save_dir)}, total epochs={self.epochs}"
        )

        epoch = self.start_epoch
        self.optimizer.zero_grad()
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            else:
                pbar = enumerate(self.train_loader)

            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                ni = i + nb * epoch

                # Warmup 学习率 & accumulate
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi,
                            [self.args.warmup_bias_lr if j == 0 else 0.0,
                             x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward + KD
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    # 1) 普通检测 loss
                    det_loss, det_loss_items = self.model(batch)

                    # 2) 预测输出：长度=3 list，元素 Tensor([B, N, 5+nc])
                    with torch.no_grad():
                        student_out = self.model.predict(batch["img"])
                        teacher_out = self.teacher.predict(batch["img"])

                    # 3) 从多尺度输出中提取分类 logits 并扁平化
                    s_list, t_list = [], []
                    nc = self.data["nc"]
                    for sr_scale, tr_scale in zip(student_out, teacher_out):
                        # 确保 Tensor 不是空
                        if isinstance(sr_scale, torch.Tensor) and sr_scale.numel() > 0:
                            # [B, N, 5+nc] → 分类部分 [B, N, nc]
                            s_cls = sr_scale[..., 5:]  # shape (B, N, nc)
                            t_cls = tr_scale[..., 5:]
                            # 扁平化成 [B*N, nc]
                            s_list.append(s_cls.reshape(-1, nc))
                            t_list.append(t_cls.reshape(-1, nc))
                    if s_list and t_list:
                        s_all = torch.cat(s_list, dim=0)  # (sum_BN, nc)
                        t_all = torch.cat(t_list, dim=0)
                        kd_l = nn.functional.kl_div(
                            nn.functional.log_softmax(s_all / self.T, dim=-1),
                            nn.functional.softmax(t_all / self.T, dim=-1),
                            reduction="batchmean"
                        ) * (self.T ** 2)
                    else:
                        kd_l = torch.tensor(0.0, device=self.device)

                    # 4) 合并总 loss & 更新日志
                    total_loss = self.alpha * det_loss + (1 - self.alpha) * kd_l
                    self.loss_items = torch.cat([det_loss_items, kd_l[None].detach().cpu()])
                    self.loss = total_loss.sum()
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1)
                        if self.tloss is not None else self.loss_items
                    )
                    LOGGER.debug(f"[KD] det_loss={det_loss.item():.4f}, kd_loss={kd_l.item():.4f}")

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stop
                    if self.args.time and (time.time() - self.train_time_start) > (self.args.time * 3600):
                        self.stop = True
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                # 进度更新 & 回调
                if RANK in {-1, 0}:
                    desc = (
                            ("%11s" * 2 + "%11.4g" * (2 + self.tloss.shape[0]))
                            % (f"{epoch + 1}/{self.epochs}", f"{self._get_memory():.3g}G",
                               *(self.tloss if self.tloss.ndim > 0 else self.tloss.unsqueeze(0)),
                               batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    pbar.set_description(desc)
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            # Epoch end
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                if self.args.val or final or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final
                if self.args.time and (time.time() - self.train_time_start) > (self.args.time * 3600):
                    self.stop = True
                if self.args.save or final:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler / memory clear / early stop...
            # … (保留原逻辑，不变)

            if self.stop:
                break
            epoch += 1

        # 结束 & teardown（同原始逻辑）
        if RANK in {-1, 0}:
            LOGGER.info(f"{epoch - self.start_epoch + 1} epochs completed.")
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        unset_deterministic()
        self.run_callbacks("teardown")

    def kd_loss(self, student_out, teacher_out):
        """
        对多尺度输出依次计算 KL 蒸馏损失，并取平均
        """
        # 如果输出是 list/tuple，就遍历
        outs_s = student_out if isinstance(student_out, (list, tuple)) else [student_out]
        outs_t = teacher_out if isinstance(teacher_out, (list, tuple)) else [teacher_out]
        assert len(outs_s) == len(outs_t), "Student/Teacher outputs count mismatch"
        loss = 0.0
        for s, t in zip(outs_s, outs_t):
            s_logits = s / self.T
            t_logits = t / self.T
            loss += nn.functional.kl_div(
                nn.functional.log_softmax(s_logits, dim=-1),
                nn.functional.softmax(t_logits, dim=-1),
                reduction='batchmean'
            )
        return loss * (self.T ** 2) / len(outs_s)
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch
    def set_model_attributes(self):
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (5 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            'kd_loss',
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        return super().auto_batch(max_num_obj)