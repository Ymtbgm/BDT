"""YOLO 技能按钮可点击状态检测器。

基于 YOLO26_skill.pt 对选中干员后弹出的技能按钮区域进行二分类检测，
判断当前技能是否处于可点击（亮起）状态，用于执行器在理论时间前 1~2 帧
技能尚未转好时自动跳帧等待。

模型输入尺寸为 100x100（与训练/导出配置一致），输出类别预期仅包含
"clickable" 一类；若检测不到目标则认为当前不可点击。
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.base.logging_utils import log_error, log_info
from core.base.paths import model


DEFAULT_INPUT_SIZE = 100
DEFAULT_CONF_THRESH = 0.5
DEFAULT_IOU_THRESH = 0.45
DEFAULT_MAX_ATTEMPTS = 10


class SkillClickDetector:
    """技能按钮可点击状态检测封装。"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
        conf_thresh: float = DEFAULT_CONF_THRESH,
        iou_thresh: float = DEFAULT_IOU_THRESH,
        threads: int = 6,
    ):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._model_path = Path(model_path) if model_path else model("YOLO", "YOLO26_skill.pt")
        self._model = None
        self.available = False
        self._last_error: Optional[str] = None
        self._clickable_class_names: Tuple[str, ...] = ("clickable",)

        if not self._model_path.exists():
            self._last_error = f"模型文件不存在: {self._model_path}"
            log_error(f"[SkillClickDetector] {self._last_error}")
            return

        try:
            import torch
            import ultralytics

            cpu_count = os.cpu_count() or 1
            torch.set_num_threads(min(threads, cpu_count))

            t0 = time.perf_counter()
            self._model = ultralytics.YOLO(str(self._model_path))
            log_info(
                f"[SkillClickDetector] 模型加载成功: {self._model_path} "
                f"(耗时={(time.perf_counter() - t0) * 1000:.1f}ms, threads={min(threads, cpu_count)})"
            )
            self.available = True
        except Exception as e:
            self._last_error = f"YOLO 初始化失败: {e}"
            log_error(f"[SkillClickDetector] {self._last_error}")
            return

        self._warm_up()

    def _warm_up(self):
        if not self.available:
            return
        try:
            dummy = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
            t0 = time.perf_counter()
            self.is_clickable(dummy)
            log_info(
                f"[SkillClickDetector] warm-up 完成 "
                f"(耗时={(time.perf_counter() - t0) * 1000:.1f}ms)"
            )
        except Exception as e:
            log_error(f"[SkillClickDetector] warm-up 失败: {e}")

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """统一转换为 3 通道 BGR。"""
        if image is None or image.size == 0:
            return image
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 3:
            return image
        # 其它通道数直接丢弃后续通道
        return image[:, :, :3]

    def _class_name(self, cls_id: int) -> str:
        """从模型 names 映射获取类别名。"""
        if self._model is None:
            return str(cls_id)
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return str(names.get(cls_id, cls_id))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    def detect(self, roi: np.ndarray) -> List[Tuple[List[int], float, str]]:
        """对技能按钮 ROI 进行检测，返回检测框、置信度、类别名列表。

        参数:
            roi: 任意通道的图像数组，建议为选中干员后技能按钮所在区域。

        返回:
            每个元素为 (bbox [x1,y1,x2,y2], conf, class_name)。
        """
        results: List[Tuple[List[int], float, str]] = []
        if not self.available or self._model is None:
            return results

        bgr = self._to_bgr(roi)
        if bgr is None or bgr.size == 0:
            return results

        try:
            preds = self._model.predict(bgr, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)
        except Exception as e:
            log_error(f"[SkillClickDetector] 推理失败: {e}")
            return results

        for result in preds:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                cls_id = int(clses[i])
                name = self._class_name(cls_id)
                conf = float(confs[i])
                if conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                results.append(([int(x1), int(y1), int(x2), int(y2)], conf, name))
        return results

    def _is_single_class_model(self) -> bool:
        """判断当前模型是否为单类别模型。"""
        if self._model is None:
            return False
        names = getattr(self._model, "names", None)
        if isinstance(names, dict):
            return len(names) == 1
        if isinstance(names, (list, tuple)):
            return len(names) == 1
        return False

    def is_clickable(self, roi: np.ndarray, debug: bool = False) -> bool:
        """判断当前 ROI 中技能按钮是否可点击。

        单类别模型下，只要检测到目标即视为可点击；
        多类别模型下，仅当类别名在可点击集合中且置信度达标才视为可点击。
        模型未加载或推理失败时默认返回 True，避免阻塞技能释放。
        """
        if not self.available:
            return True
        detections = self.detect(roi)
        single_class = self._is_single_class_model()
        for box, conf, name in detections:
            if single_class or name in self._clickable_class_names:
                if debug:
                    print(
                        f"[SkillClickDetector] 检测到可点击技能: box={box}, "
                        f"conf={conf:.3f}, class={name}"
                    )
                return True
        if debug:
            if detections:
                print(
                    f"[SkillClickDetector] 检测到 {len(detections)} 个框，但均不视为可点击: "
                    + ", ".join(f"{name}({conf:.3f})" for _, conf, name in detections)
                )
            else:
                print("[SkillClickDetector] 未检测到技能按钮")
        return False

    def set_clickable_class_names(self, names: Tuple[str, ...]):
        """允许外部调整被视为"可点击"的类别名集合。"""
        self._clickable_class_names = names
