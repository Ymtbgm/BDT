"""YOLO 数量角标检测器。

基于 YOLO26_quantity.pt 对 2560×65 的二值化数量条进行滑动窗口切片检测，
输出候选数量角标框（x1,y1,x2,y2）及置信度，供 OfflineResolver 替代 Paddle
的 detect_text 使用。
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.base.logging_utils import log_error, log_info
from core.base.paths import model


# 与 YOLO 检测模型部署指南保持一致
SLICE_W = 640
STRIDE = 544
FULL_W = 2560
FULL_H = 65
DEFAULT_CONF_THRESH = 0.5
DEFAULT_IOU_THRESH = 0.5


class QuantityBadgeDetector:
    """YOLO 数量角标检测封装。"""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        conf_thresh: float = DEFAULT_CONF_THRESH,
        iou_thresh: float = DEFAULT_IOU_THRESH,
        threads: int = 6,
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._model_path = Path(model_path) if model_path else model("YOLO", "YOLO26_quantity.pt")
        self._model = None
        self.available = False
        self._last_error: Optional[str] = None

        if not self._model_path.exists():
            self._last_error = f"模型文件不存在: {self._model_path}"
            log_error(f"[QuantityBadgeDetector] {self._last_error}")
            return

        try:
            import torch
            import ultralytics

            # 限制 PyTorch 线程数，避免与 ONNX Runtime 在 CPU 上争抢
            cpu_count = os.cpu_count() or 1
            torch.set_num_threads(min(threads, cpu_count))

            t0 = time.perf_counter()
            self._model = ultralytics.YOLO(str(self._model_path))
            log_info(
                f"[QuantityBadgeDetector] 模型加载成功: {self._model_path} "
                f"(耗时={(time.perf_counter() - t0) * 1000:.1f}ms, threads={min(threads, cpu_count)})"
            )
            self.available = True
        except Exception as e:
            self._last_error = f"YOLO 初始化失败: {e}"
            log_error(f"[QuantityBadgeDetector] {self._last_error}")
            try:
                import traceback
                from pathlib import Path
                err_path = Path("debug") / "yolo_init_error.log"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(
                    f"{self._last_error}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
            except Exception:
                pass
            return

        # 一次空推理完成 warm-up，避免首次真实检测触发 JIT 编译
        self._warm_up()

    @staticmethod
    def _compute_slice_starts(full_w: int, slice_w: int, stride: int) -> List[int]:
        starts = list(range(0, full_w - slice_w + 1, stride))
        last_start = full_w - slice_w
        if not starts or starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float,
    ) -> List[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def _warm_up(self):
        if not self.available:
            return
        try:
            dummy = np.full((FULL_H, FULL_W, 3), 255, dtype=np.uint8)
            t0 = time.perf_counter()
            self.detect(dummy)
            log_info(
                f"[QuantityBadgeDetector] warm-up 完成 "
                f"(耗时={(time.perf_counter() - t0) * 1000:.1f}ms)"
            )
        except Exception as e:
            log_error(f"[QuantityBadgeDetector] warm-up 失败: {e}")

    def detect(self, strip: np.ndarray) -> List[Tuple[List[Tuple[int, int]], float]]:
        """在 2560×65 数量条上检测 X+数字 角标框。

        参数:
            strip: 3 通道 BGR 或灰度图，建议传入 _preprocess_quantity_strip 二值化结果。

        返回:
            每个元素为 (四边形框, 检测置信度)，框为 4 个 (x, y) 点。
        """
        if not self.available or self._model is None:
            return []

        if strip is None or strip.size == 0:
            return []

        # 统一为 3 通道 BGR
        if len(strip.shape) == 2:
            proc = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        elif strip.shape[2] == 4:
            proc = cv2.cvtColor(strip, cv2.COLOR_BGRA2BGR)
        else:
            proc = strip

        h, w = proc.shape[:2]
        if w != FULL_W or h != FULL_H:
            # 非标准尺寸直接整图推理（兜底）
            results = self._model.predict(proc, conf=self.conf_thresh, verbose=False)
            return self._extract_boxes(results, x_offset=0, img_w=w)

        slice_starts = self._compute_slice_starts(FULL_W, SLICE_W, STRIDE)
        slices = [proc[:, sx : sx + SLICE_W] for sx in slice_starts]

        # 批量推理： Ultralytics 对 list 输入会自动组 batch，
        # 5 张 640×65 切片一次过模型比逐张串行快约 3 倍。
        results = self._model.predict(slices, conf=self.conf_thresh, verbose=False)

        all_xyxy: List[np.ndarray] = []
        all_scores: List[float] = []
        for sx, result in zip(slice_starts, results):
            boxes, scores = self._extract_boxes(
                [result], x_offset=sx, img_w=FULL_W, return_xyxy=True
            )
            all_xyxy.extend(boxes)
            all_scores.extend(scores)

        if not all_xyxy:
            return []

        boxes_arr = np.array(all_xyxy, dtype=np.float32)
        scores_arr = np.array(all_scores, dtype=np.float32)
        keep = self._nms(boxes_arr, scores_arr, self.iou_thresh)

        output: List[Tuple[List[Tuple[int, int]], float]] = []
        for idx in keep:
            x1, y1, x2, y2 = boxes_arr[idx]
            quad = [
                (int(x1), int(y1)),
                (int(x2), int(y1)),
                (int(x2), int(y2)),
                (int(x1), int(y2)),
            ]
            output.append((quad, float(scores_arr[idx])))
        return output

    def _extract_boxes(
        self,
        results,
        x_offset: int = 0,
        img_w: int = FULL_W,
        return_xyxy: bool = False,
    ):
        """从 ultralytics 结果中提取框。"""
        boxes: List[np.ndarray] = []
        scores: List[float] = []
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                x1 = max(0, min(img_w, x1 + x_offset))
                x2 = max(0, min(img_w, x2 + x_offset))
                y1 = max(0, y2 if y1 > y2 else y1)
                y2 = max(y1, y2 if y2 > y1 else y1)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(np.array([x1, y1, x2, y2], dtype=np.float32))
                scores.append(float(confs[i]))
        if return_xyxy:
            return boxes, scores
        # 转换为四边形格式
        output: List[Tuple[List[Tuple[int, int]], float]] = []
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            quad = [
                (int(x1), int(y1)),
                (int(x2), int(y1)),
                (int(x2), int(y2)),
                (int(x1), int(y2)),
            ]
            output.append((quad, score))
        return output
