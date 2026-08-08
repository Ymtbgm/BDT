from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.base.logging_utils import log_error, log_info
from core.base.onnx_utils import create_session_options, get_onnx_providers
from core.base.paths import model


class AvatarMatcherBase(ABC):
    """头像匹配器基类，统一提供分数矩阵与贪心分配接口。"""

    @abstractmethod
    def compute_score_matrix(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
    ) -> Dict[str, Dict[int, float]]:
        """返回 {template_name: {cell_index: score}}。"""
        ...

    @abstractmethod
    def classify_cells(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
        min_score: float = 0.5,
    ) -> Dict[str, int]:
        """多模板对多单元格贪心最优分配，返回 {name: cell_index}。"""
        ...

    def set_template_cache(self, templates: Dict[str, np.ndarray]) -> None:
        """预计算并缓存模板特征（可选）。子类可选择实现以加速重复匹配。"""
        pass


class AvatarMatcher(AvatarMatcherBase):
    """基于 cv2.matchTemplate 的头像匹配工具。

    所有图像统一转换为灰度并缩放到固定尺寸（默认 64x64）后匹配。
    支持 CLAHE 对比度增强、圆形掩膜（屏蔽边框与角上 UI）以及多尺度匹配。
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (64, 64),
        use_clahe: bool = True,
        use_circular_mask: bool = True,
        scales: Tuple[float, ...] = (0.92, 0.96, 1.0, 1.04, 1.08),
    ):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.use_circular_mask = use_circular_mask
        self.scales = scales
        if use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        else:
            self._clahe = None
        self._mask = self._build_circular_mask(target_size) if use_circular_mask else None
        self._cached_norm_templates: Optional[Dict[str, np.ndarray]] = None
        self._cached_template_keys: Optional[List[str]] = None

    @staticmethod
    def _build_circular_mask(size: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(size, dtype=np.uint8)
        cx, cy = size[0] // 2, size[1] // 2
        radius = min(cx, cy)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """转换为灰度、CLAHE 增强、缩放并可选应用圆形掩膜。"""
        if image is None or image.size == 0:
            out = np.zeros(self.target_size, dtype=np.uint8)
        elif len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            out = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_CUBIC)
        elif len(image.shape) == 2:
            out = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        else:
            out = np.zeros(self.target_size, dtype=np.uint8)

        if self._clahe is not None:
            out = self._clahe.apply(out)
        if self._mask is not None:
            out = cv2.bitwise_and(out, out, mask=self._mask)
        return out

    def _match_single_scale(
        self,
        template_gray: np.ndarray,
        cell_gray: np.ndarray,
    ) -> float:
        """在单尺度下返回最佳匹配分数。"""
        if template_gray.shape[0] > cell_gray.shape[0] or template_gray.shape[1] > cell_gray.shape[1]:
            return -1.0
        result = cv2.matchTemplate(cell_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    def set_template_cache(self, templates: Dict[str, np.ndarray]) -> None:
        """预计算并缓存归一化后的模板，避免每次匹配重复做灰度/缩放/掩膜。"""
        if not templates:
            self._cached_norm_templates = None
            self._cached_template_keys = None
            return
        self._cached_template_keys = list(templates.keys())
        self._cached_norm_templates = {
            name: self._normalize(img) for name, img in templates.items()
        }

    def compute_score_matrix(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
    ) -> Dict[str, Dict[int, float]]:
        """计算多尺度下每个模板对每个单元格的匹配分数矩阵。"""
        if not templates or not cells:
            return {}

        if self._cached_norm_templates is not None and list(templates.keys()) == self._cached_template_keys:
            norm_templates = self._cached_norm_templates
        else:
            norm_templates = {name: self._normalize(img) for name, img in templates.items()}
        norm_cells = [self._normalize(img) for img in cells]

        matrix: Dict[str, Dict[int, float]] = {}
        for name, t in norm_templates.items():
            cell_scores: Dict[int, float] = {}
            for cell_idx, cell in enumerate(norm_cells):
                best_score = -1.0
                for scale in self.scales:
                    size = (max(8, int(t.shape[1] * scale)), max(8, int(t.shape[0] * scale)))
                    scaled = cv2.resize(t, size, interpolation=cv2.INTER_CUBIC)
                    score = self._match_single_scale(scaled, cell)
                    if score > best_score:
                        best_score = score
                cell_scores[cell_idx] = best_score
            matrix[name] = cell_scores
        return matrix

    def classify_cells(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
        min_score: float = 0.5,
    ) -> Dict[str, int]:
        """多模板对多单元格贪心最优分配（支持多尺度）。"""
        matrix = self.compute_score_matrix(templates, cells)
        scores: List[Tuple[float, str, int]] = []
        for name, cell_scores in matrix.items():
            for idx, score in cell_scores.items():
                scores.append((score, name, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        assigned_names: set = set()
        assigned_cells: set = set()
        result: Dict[str, int] = {}
        for score, name, idx in scores:
            if score < min_score:
                continue
            if name in assigned_names or idx in assigned_cells:
                continue
            result[name] = idx
            assigned_names.add(name)
            assigned_cells.add(idx)
        return result


class ResNetAvatarMatcher(AvatarMatcherBase):
    """基于 ResNet18 嵌入向量的头像匹配工具。

    使用 torchvision 预训练 ResNet18 提取 512 维特征，按余弦相似度匹配。
    对截图风格差异（编队界面 vs 部署栏图标、不同光照/缩放）比模板匹配更鲁棒。
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        device: Optional[str] = None,
        min_score: float = 0.70,
        input_size: int = 224,
    ):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_score = min_score
        self.input_size = input_size

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            base_model = models.resnet18(weights=weights)
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            base_model = models.resnet34(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            base_model = models.resnet50(weights=weights)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 去掉最后的分类层，仅保留特征向量（用于头像匹配）
        self.full_model = base_model
        self.full_model.eval().to(self.device)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval().to(self.device)

        resize_size = int(input_size * 256 / 224)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._cached_template_names: Optional[List[str]] = None
        self._cached_template_embs: Optional[np.ndarray] = None

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        """统一转换为 RGB uint8。"""
        if image is None or image.size == 0:
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

    def _extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """批量提取特征向量，返回 (N, D) numpy 数组。"""
        tensors = []
        for img in images:
            rgb = self._to_rgb(img)
            tensor = self.transform(rgb)
            tensors.append(tensor)
        batch = self._torch.stack(tensors).to(self.device)
        with self._torch.no_grad():
            features = self.model(batch)
        # features shape: (N, D, 1, 1)
        return features.squeeze().cpu().numpy()

    def export_onnx(self, output_path: Path) -> Path:
        """导出完整 ResNet18，再剪掉最后的 fc，得到 512 维特征 ONNX（兼顾速度和精度）。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.full_model.eval()
        dummy = self._torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
        self._torch.onnx.export(
            self.full_model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            opset_version=11,
            dynamo=False,
        )
        return _trim_resnet18_fc(output_path)

    def set_template_cache(self, templates: Dict[str, np.ndarray]) -> None:
        """预计算并 L2 归一化模板特征，避免每次匹配重复提取。"""
        if not templates:
            self._cached_template_names = None
            self._cached_template_embs = None
            return
        names = list(templates.keys())
        embs = self._extract_batch([templates[n] for n in names])
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        self._cached_template_names = names
        self._cached_template_embs = embs

    def compute_score_matrix(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
    ) -> Dict[str, Dict[int, float]]:
        """计算模板与单元格间的余弦相似度矩阵。"""
        if not templates or not cells:
            return {}

        names = list(templates.keys())
        if self._cached_template_embs is not None and names == self._cached_template_names:
            template_embs = self._cached_template_embs
        else:
            template_embs = self._extract_batch([templates[n] for n in names])
            if template_embs.ndim == 1:
                template_embs = template_embs.reshape(1, -1)
            template_embs = template_embs / (np.linalg.norm(template_embs, axis=1, keepdims=True) + 1e-8)

        cell_embs = self._extract_batch(cells)

        if cell_embs.ndim == 1:
            cell_embs = cell_embs.reshape(1, -1)

        cell_embs = cell_embs / (np.linalg.norm(cell_embs, axis=1, keepdims=True) + 1e-8)

        sim_matrix = template_embs @ cell_embs.T  # (N, M)

        matrix: Dict[str, Dict[int, float]] = {}
        for i, name in enumerate(names):
            matrix[name] = {j: float(sim_matrix[i, j]) for j in range(len(cells))}
        return matrix

    def classify_cells(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
        min_score: Optional[float] = None,
    ) -> Dict[str, int]:
        """多模板对多单元格贪心最优分配（余弦相似度）。"""
        matrix = self.compute_score_matrix(templates, cells)
        threshold = self.min_score if min_score is None else min_score

        scores: List[Tuple[float, str, int]] = []
        for name, cell_scores in matrix.items():
            for idx, score in cell_scores.items():
                scores.append((score, name, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        assigned_names: set = set()
        assigned_cells: set = set()
        result: Dict[str, int] = {}
        for score, name, idx in scores:
            if score < threshold:
                continue
            if name in assigned_names or idx in assigned_cells:
                continue
            result[name] = idx
            assigned_names.add(name)
            assigned_cells.add(idx)
        return result


def _default_onnx_path(input_size: int) -> Path:
    return model("ResNet", f"resnet18_avatar_matcher_{input_size}.onnx")


def _trim_resnet18_fc(onnx_path: Path) -> Path:
    """把完整 ResNet18 ONNX 最后的 fc 层剪掉，输出 512 维特征（保留 avgpool + Flatten）。"""
    import onnx
    from onnx import helper

    onnx_path = Path(onnx_path)
    model = onnx.load(str(onnx_path))
    graph = model.graph

    flatten_node = None
    gemm_node = None
    for node in graph.node:
        if node.op_type == "Flatten":
            flatten_node = node
        if node.op_type == "Gemm" and (node.name == "/fc/Gemm" or "fc" in node.name):
            gemm_node = node

    if flatten_node is None or gemm_node is None:
        return onnx_path

    graph.node.remove(gemm_node)
    for init in list(graph.initializer):
        if init.name in list(gemm_node.input):
            graph.initializer.remove(init)

    del graph.output[:]
    output_name = flatten_node.output[0]
    graph.output.append(helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 512]))

    onnx.save(model, str(onnx_path))
    return onnx_path


def ensure_resnet18_onnx(input_size: int = 224, output_path: Optional[Path] = None) -> Path:
    """确保 ONNX 模型存在；不存在则导出。返回最终路径。"""
    output_path = output_path or _default_onnx_path(input_size)
    output_path = Path(output_path)
    if output_path.exists():
        return output_path
    matcher = ResNetAvatarMatcher(input_size=input_size)
    matcher.export_onnx(output_path)
    log_info(f"[头像匹配] ONNX 模型已导出: {output_path}")
    return output_path


def ensure_mobilenetv4_onnx(
    model_name: str = "mobilenetv4_conv_small",
    input_size: int = 224,
    output_path: Optional[Path] = None,
) -> Path:
    """确保 MobileNetV4 ONNX 模型存在；不存在则通过 timm + ModelScope 下载并导出。

    此函数仅在开发/导出阶段使用，打包给用户时只分发 .onnx 文件即可，不需要 timm。
    """
    output_path = output_path or model("MoblienetV4", f"{model_name}_avatar_matcher_{input_size}.onnx")
    output_path = Path(output_path)
    if output_path.exists():
        return output_path

    import timm
    import torch
    from modelscope import snapshot_download

    cache_dir = snapshot_download(f"timm/{model_name}.e2400_r224_in1k")
    bin_path = Path(cache_dir) / "pytorch_model.bin"

    m = timm.create_model(model_name, pretrained=False, num_classes=0)
    m.load_state_dict(torch.load(str(bin_path), map_location="cpu"), strict=False)
    m.eval()

    dummy = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(
        m,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamo=False,
    )
    log_info(f"[头像匹配] MobileNetV4 ONNX 模型已导出: {output_path}")
    return output_path


class ONNXFeatureAvatarMatcher(AvatarMatcherBase):
    """基于 ONNX 特征提取器的头像匹配工具。

    预处理使用 ImageNet 归一化，推理使用 onnxruntime，不依赖 torch/torchvision。
    兼容 ResNet18、MobileNetV4 等导出为固定 batch=1 的特征向量 ONNX 模型。
    自动优先使用 DirectML 轻量 GPU provider，不可用则回退 CPU。
    """

    def __init__(
        self,
        model_path: Path,
        input_size: int = 224,
        min_score: float = 0.70,
    ):
        import onnxruntime as ort

        self.input_size = input_size
        self.min_score = min_score

        sess_options = create_session_options()
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=get_onnx_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.providers = self.session.get_providers()
        log_info(f"[头像匹配] ONNX 后端: {self.providers}")

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        self._cached_template_names: Optional[List[str]] = None
        self._cached_template_embs: Optional[np.ndarray] = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """将任意 OpenCV 图像预处理为 CHW、ImageNet 归一化的 float32 数组。"""
        if image is None or image.size == 0:
            rgb = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        elif len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

        h, w = rgb.shape[:2]
        resize_size = int(self.input_size * 256 / 224)
        scale = resize_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        start_h = (new_h - self.input_size) // 2
        start_w = (new_w - self.input_size) // 2
        cropped = resized[start_h:start_h + self.input_size, start_w:start_w + self.input_size]

        arr = cropped.astype(np.float32) / 255.0
        arr = (arr - self._mean.reshape(3, 1, 1).T) / self._std.reshape(3, 1, 1).T
        arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        # 当前导出的 ONNX 是固定 batch=1 以获得最快推理速度；
        # batch > 1 时退化成逐张推理，避免开启动态轴带来的性能损失。
        if len(images) == 1:
            batch = np.stack([self._preprocess(images[0])], axis=0)
            outputs = self.session.run(None, {self.input_name: batch})[0]
            return outputs.reshape(outputs.shape[0], -1)
        feats = [self._extract_batch([img]) for img in images]
        return np.concatenate(feats, axis=0)

    def set_template_cache(self, templates: Dict[str, np.ndarray]) -> None:
        """预计算并 L2 归一化模板特征，避免每次匹配重复提取。"""
        if not templates:
            self._cached_template_names = None
            self._cached_template_embs = None
            return
        names = list(templates.keys())
        embs = self._extract_batch([templates[n] for n in names])
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        self._cached_template_names = names
        self._cached_template_embs = embs

    def compute_score_matrix(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
    ) -> Dict[str, Dict[int, float]]:
        """计算模板与单元格间的余弦相似度矩阵。"""
        if not templates or not cells:
            return {}

        names = list(templates.keys())
        if self._cached_template_embs is not None and names == self._cached_template_names:
            template_embs = self._cached_template_embs
        else:
            template_embs = self._extract_batch([templates[n] for n in names])
            if template_embs.ndim == 1:
                template_embs = template_embs.reshape(1, -1)
            template_embs = template_embs / (np.linalg.norm(template_embs, axis=1, keepdims=True) + 1e-8)

        cell_embs = self._extract_batch(cells)

        if cell_embs.ndim == 1:
            cell_embs = cell_embs.reshape(1, -1)

        cell_embs = cell_embs / (np.linalg.norm(cell_embs, axis=1, keepdims=True) + 1e-8)

        sim_matrix = template_embs @ cell_embs.T

        matrix: Dict[str, Dict[int, float]] = {}
        for i, name in enumerate(names):
            matrix[name] = {j: float(sim_matrix[i, j]) for j in range(len(cells))}
        return matrix

    def classify_cells(
        self,
        templates: Dict[str, np.ndarray],
        cells: List[np.ndarray],
        min_score: Optional[float] = None,
    ) -> Dict[str, int]:
        """多模板对多单元格贪心最优分配（余弦相似度）。"""
        matrix = self.compute_score_matrix(templates, cells)
        threshold = self.min_score if min_score is None else min_score

        scores: List[Tuple[float, str, int]] = []
        for name, cell_scores in matrix.items():
            for idx, score in cell_scores.items():
                scores.append((score, name, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        assigned_names: set = set()
        assigned_cells: set = set()
        result: Dict[str, int] = {}
        for score, name, idx in scores:
            if score < threshold:
                continue
            if name in assigned_names or idx in assigned_cells:
                continue
            result[name] = idx
            assigned_names.add(name)
            assigned_cells.add(idx)
        return result


# 保留旧名称兼容性
ONNXResNetAvatarMatcher = ONNXFeatureAvatarMatcher


class LogoMiniCNNMatcher:
    """基于自训练小型 CNN 的头顶 Logo 二分类器（ONNX Runtime）。

    输入 64x64 RGB，输出 active / inactive 的 softmax 概率。
    预处理与训练时一致：直接 resize 到 64x64，ImageNet 归一化。
    """

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.input_size = 64
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        sess_options = create_session_options()
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=get_onnx_providers()
        )
        self.input_name = self.session.get_inputs()[0].name

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        """统一转换为 RGB uint8。"""
        if image is None or image.size == 0:
            return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """返回 (1, 3, 64, 64) 的 float32 数组。"""
        rgb = self._to_rgb(image)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self._mean.reshape(3, 1, 1).T) / self._std.reshape(3, 1, 1).T
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-8)

    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """返回 {'active': prob, 'inactive': prob}。"""
        x = self._preprocess(image)
        logits = self.session.run(None, {self.input_name: x})[0]
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        probs = self._softmax(logits)
        return {"active": float(probs[0, 0]), "inactive": float(probs[0, 1])}


class LogoMiniCNNEmbedder:
    """基于自训练小型 CNN 的 128 维特征提取器（ONNX Runtime）。

    输入 64x64 RGB，输出 128 维特征（未做 L2 归一化），供头顶 Logo 相似度匹配使用。
    预处理与 LogoMiniCNNMatcher / 训练时保持一致。
    """

    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.input_size = 64
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

        sess_options = create_session_options()
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=get_onnx_providers()
        )
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """统一转换为 RGB uint8。"""
        if image is None or image.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """返回 (3, 64, 64) 的 float32 数组。"""
        rgb = self._to_rgb(image)
        resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - self._mean.reshape(3, 1, 1).T) / self._std.reshape(3, 1, 1).T
        arr = np.transpose(arr, (2, 0, 1))
        return arr.astype(np.float32)

    def _extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """对一批图像提取 128 维特征，返回 (N, 128)。"""
        if not images:
            return np.zeros((0, 128), dtype=np.float32)
        tensors = np.stack([self._preprocess(img) for img in images], axis=0)
        feats = self.session.run(None, {self.input_name: tensors})[0]
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        return feats


_resnet_matcher_cache: Optional[ResNetAvatarMatcher] = None


def preload_resnet(device: Optional[str] = None) -> Optional[ResNetAvatarMatcher]:
    """预加载 ResNet 匹配器并缓存，便于在 OCR 初始化阶段一并完成。"""
    global _resnet_matcher_cache
    if _resnet_matcher_cache is not None:
        return _resnet_matcher_cache
    try:
        _resnet_matcher_cache = ResNetAvatarMatcher(device=device)
        log_info("[头像匹配] ResNet18 预加载完成")
        return _resnet_matcher_cache
    except Exception as e:
        log_info(f"[头像匹配] ResNet18 预加载失败: {e}")
        return None


def create_avatar_matcher(
    prefer_resnet: bool = True,
    input_size: int = 224,
    use_onnx: bool = False,
    model_name: str = "resnet18",
) -> AvatarMatcherBase:
    """创建头像匹配器。

    优先顺序：
      1. ONNX 模型（当 use_onnx=True 时；按 model_name 选择 ResNet18 或 MobileNetV4）
      2. PyTorch ResNet18（缓存或新建）
      3. 多尺度模板匹配

    Args:
        model_name: 特征提取器名称，支持 "resnet18" / "mobilenetv4_conv_small" 等。
                    仅在 use_onnx=True 时生效。
    """
    global _resnet_matcher_cache
    if prefer_resnet:
        if use_onnx:
            if model_name.startswith("mobilenetv4"):
                try:
                    onnx_path = ensure_mobilenetv4_onnx(model_name=model_name, input_size=input_size)
                    return ONNXFeatureAvatarMatcher(onnx_path, input_size=input_size)
                except Exception as e:
                    log_info(f"[头像匹配] ONNX {model_name} 初始化失败，尝试 ResNet18: {e}")
            try:
                onnx_path = ensure_resnet18_onnx(input_size)
                return ONNXFeatureAvatarMatcher(onnx_path, input_size=input_size)
            except Exception as e:
                log_info(f"[头像匹配] ONNX ResNet18 初始化失败，尝试 PyTorch: {e}")
        if _resnet_matcher_cache is not None:
            return _resnet_matcher_cache
        try:
            return ResNetAvatarMatcher(input_size=input_size)
        except Exception as e:
            log_info(f"[头像匹配] ResNet 初始化失败，回退到模板匹配: {e}")
    return AvatarMatcher()
