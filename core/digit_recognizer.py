"""轻量 ONNX 数字分类器封装。

加载 core/resource/models/ 下的两个分类模型：
- num_digit_model.onnx : 费用数字 0~99，输入 (1,1,72,106)
- X_num_model.onnx     : 数量角标 X0~X30，输入 (1,1,130,214)

预处理约定：
- 调用方应先使用项目现有预处理（cost_recognition.preprocess_cost_image_inv、
  resolver._preprocess_quantity_strip invert=True）得到黑字白底图；
- 本模块仅做最后的归一化与张量排布，若尺寸不符则自动 resize/反色兜底。
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.logging_utils import log_error, log_info
from core.onnx_utils import create_session_options, get_onnx_providers


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "core" / "resource" / "models"

# 模型配置
_COST_MODEL_PATH = MODEL_DIR / "num_digit_model.onnx"
_COST_MAP_PATH = MODEL_DIR / "num_class_to_idx.json"
_COST_INPUT_SIZE = (106, 72)  # (W, H)

_QUANTITY_MODEL_PATH = MODEL_DIR / "X_num_model.onnx"
_QUANTITY_MAP_PATH = MODEL_DIR / "X_num_class_to_idx.json"
_QUANTITY_INPUT_SIZE = (214, 130)  # (W, H)


class DigitRecognizer:
    """基于 ONNX 的费用/数量数字分类器。"""

    def __init__(
        self,
        use_gpu: bool = False,
    ):
        self._cost_session = None
        self._quantity_session = None
        self._cost_idx_to_class: Dict[int, str] = {}
        self._quantity_idx_to_class: Dict[int, str] = {}
        self._cost_input_name: Optional[str] = None
        self._quantity_input_name: Optional[str] = None
        self._providers: List[str] = []

        try:
            self._load_models(use_gpu=use_gpu)
            self.available = True
            self._warm_up()
        except Exception as e:
            log_error(f"[DigitRecognizer] 模型加载失败，将不可用: {e}")
            self.available = False

    def _warm_up(self):
        if not self.available:
            return
        try:
            t0 = time.perf_counter()
            cost_dummy = self._normalize(
                np.full((_COST_INPUT_SIZE[1], _COST_INPUT_SIZE[0]), 255, dtype=np.uint8)
            )
            self._cost_session.run(None, {self._cost_input_name: cost_dummy})
            qty_dummy = self._normalize(
                np.full(
                    (_QUANTITY_INPUT_SIZE[1], _QUANTITY_INPUT_SIZE[0]), 255, dtype=np.uint8
                )
            )
            self._quantity_session.run(None, {self._quantity_input_name: qty_dummy})
            log_info(
                f"[DigitRecognizer] warm-up 完成 "
                f"(耗时={(time.perf_counter() - t0) * 1000:.1f}ms)"
            )
        except Exception as e:
            log_error(f"[DigitRecognizer] warm-up 失败: {e}")

    def _load_models(self, use_gpu: bool) -> None:
        import onnxruntime as ort

        providers = get_onnx_providers(prefer="directml" if use_gpu else "cpu")
        self._providers = providers

        sess_options = create_session_options()

        # 费用模型
        if not _COST_MODEL_PATH.exists() or not _COST_MAP_PATH.exists():
            raise FileNotFoundError(
                f"费用模型文件缺失: {_COST_MODEL_PATH}, {_COST_MAP_PATH}"
            )
        self._cost_session = ort.InferenceSession(
            str(_COST_MODEL_PATH), sess_options=sess_options, providers=providers
        )
        self._cost_input_name = self._cost_session.get_inputs()[0].name
        with _COST_MAP_PATH.open("r", encoding="utf-8") as f:
            cost_class_to_idx = json.load(f)
        self._cost_idx_to_class = {v: k for k, v in cost_class_to_idx.items()}

        # 数量模型
        if not _QUANTITY_MODEL_PATH.exists() or not _QUANTITY_MAP_PATH.exists():
            raise FileNotFoundError(
                f"数量模型文件缺失: {_QUANTITY_MODEL_PATH}, {_QUANTITY_MAP_PATH}"
            )
        self._quantity_session = ort.InferenceSession(
            str(_QUANTITY_MODEL_PATH), sess_options=sess_options, providers=providers
        )
        self._quantity_input_name = self._quantity_session.get_inputs()[0].name
        with _QUANTITY_MAP_PATH.open("r", encoding="utf-8") as f:
            quantity_class_to_idx = json.load(f)
        self._quantity_idx_to_class = {
            v: k for k, v in quantity_class_to_idx.items()
        }

        log_info(
            f"[DigitRecognizer] 模型加载成功，provider={self._providers}, "
            f"cost_classes={len(self._cost_idx_to_class)}, "
            f"quantity_classes={len(self._quantity_idx_to_class)}"
        )

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """统一转换为单通道灰度。"""
        if image is None or image.size == 0:
            return image
        if len(image.shape) == 2:
            return image
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _ensure_black_on_white(
        gray: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """确保黑字白底并 resize 到目标尺寸 (W, H)。"""
        if gray is None or gray.size == 0:
            return np.full((target_size[1], target_size[0]), 255, dtype=np.uint8)

        # 极性判断：均值 < 127 视为白字黑底，需要反色
        if float(gray.mean()) < 127.0:
            gray = 255 - gray

        # 尺寸兜底
        if gray.shape[1] != target_size[0] or gray.shape[0] != target_size[1]:
            gray = cv2.resize(
                gray,
                target_size,
                interpolation=cv2.INTER_LANCZOS4,
            )
        return gray

    def _maybe_thicken(
        self, gray: np.ndarray, density_threshold: float = 0.07
    ) -> np.ndarray:
        """thick7 智能加粗：仅对右下角细字区域做形态学膨胀，缓解笔画过细问题。

        触发条件（同时满足）：
        - 右下角 ROI（图像右侧 65%、下方 75%）存在前景；
        - 文字高度比 < 90%；
        - ROI 笔画密度 < density_threshold。
        """
        if gray is None or gray.size == 0:
            return gray

        h, w = gray.shape[:2]
        roi_x0, roi_y0 = int(w * 0.35), int(h * 0.25)
        roi = gray[roi_y0:, roi_x0:]

        # 二值化：黑字白底图中，< 200 为前景
        _, binary_inv = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY_INV)
        fg_mask = binary_inv > 0
        if fg_mask.sum() == 0:
            return gray

        ys, _ = np.where(fg_mask)
        fg_height_ratio = (ys.max() - ys.min() + 1) / h
        roi_density = fg_mask.sum() / fg_mask.size

        if fg_height_ratio >= 0.90 or roi_density >= density_threshold:
            return gray

        # 3x3 膨胀，等价于 PIL ImageFilter.MaxFilter(3)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(binary_inv, kernel, iterations=1)

        thickened_roi = roi.copy()
        thickened_roi[dilated > 0] = 0

        result = gray.copy()
        result[roi_y0:, roi_x0:] = thickened_roi
        log_info(
            f"[DigitRecognizer] thick7 加粗触发: "
            f"fg_height_ratio={fg_height_ratio:.2f}, roi_density={roi_density:.3f}"
        )
        return result

    @staticmethod
    def _normalize(gray: np.ndarray) -> np.ndarray:
        """将 0~255 灰度图归一化到 [-1, 1] 并转 NCHW。"""
        arr = gray.astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        return arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _predict(
        self,
        image: np.ndarray,
        session,
        input_name: str,
        idx_to_class: Dict[int, str],
        target_size: Tuple[int, int],
        thicken: bool = False,
    ) -> Optional[Tuple[str, float]]:
        if session is None or not input_name:
            return None

        t0 = time.perf_counter()
        gray = self._to_gray(image)
        gray = self._ensure_black_on_white(gray, target_size)
        if thicken:
            gray = self._maybe_thicken(gray)
        tensor = self._normalize(gray)
        pre_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        logits = session.run(None, {input_name: tensor})[0][0]
        inf_ms = (time.perf_counter() - t1) * 1000

        probs = self._softmax(logits)
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        pred_class = idx_to_class.get(pred_idx)
        log_info(
            f"[DigitRecognizer] _predict pre={pre_ms:.2f}ms inf={inf_ms:.2f}ms "
            f"shape={image.shape} target={target_size} thicken={thicken}"
        )
        return (pred_class, conf) if pred_class is not None else None

    def predict_cost(self, image: np.ndarray) -> Optional[Tuple[int, float]]:
        """识别费用数字，返回 (value, confidence)。

        输入建议为 `preprocess_cost_image_inv` 后的黑字白底 106×72 图。
        """
        result = self._predict(
            image,
            self._cost_session,
            self._cost_input_name or "",
            self._cost_idx_to_class,
            _COST_INPUT_SIZE,
            thicken=False,
        )
        if result is None:
            return None
        class_name, conf = result
        try:
            value = int(class_name)
        except ValueError:
            return None
        if not (0 <= value <= 99):
            return None
        return value, conf

    def predict_quantity(self, image: np.ndarray) -> Optional[Tuple[int, float]]:
        """识别数量角标，返回数字部分 (value, confidence)。

        输入建议为 `_preprocess_quantity_strip(invert=True)` 后的黑字白底 214×130 图。
        本方法会先对右下角细字区域做 thick7 智能加粗，提升过细笔画的识别率。
        """
        result = self._predict(
            image,
            self._quantity_session,
            self._quantity_input_name or "",
            self._quantity_idx_to_class,
            _QUANTITY_INPUT_SIZE,
            thicken=True,
        )
        if result is None:
            return None
        class_name, conf = result
        m = re.match(r"^[xX×*]?(\d+)$", class_name)
        if not m:
            return None
        try:
            value = int(m.group(1))
        except ValueError:
            return None
        if not (0 <= value <= 30):
            return None
        return value, conf
