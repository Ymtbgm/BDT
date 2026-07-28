import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


class OCREngine:
    def __init__(
        self,
        use_gpu: bool = False,
        debug: bool = False,
        engine: Optional[str] = None,
        model_size: str = "mobile",  # "mobile" 速度快，"server" 精度高
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
    ):
        self.debug = debug
        self._model_size = model_size
        # engine=None 让 PaddleOCR 自行选择（默认 paddle_static，最快）
        # 如果当前环境 paddle_static 崩溃，可显式传入 engine="transformers"
        # engine="paddlex_onnx" 使用 PaddleX create_pipeline(..., engine="onnxruntime")，
        # 推理速度通常比 PaddleOCR transformers 后端更快。
        # 当前环境下 Paddle 原生静态推理需要 CUDA，无 CUDA 时直接走 PaddleX ONNX Runtime。
        if engine in ("paddle", "paddle_static"):
            print(f"[OCR] 引擎 '{engine}' 在当前无 CUDA 环境下已映射为 PaddleX ONNX Runtime")
            engine = "paddlex_onnx"
        self._user_engine = engine
        self._det_model_dir = det_model_dir
        self._rec_model_dir = rec_model_dir
        self._ocr = None
        self._backend = "unknown"
        self.device = "unknown"
        self._init_ocr(use_gpu=use_gpu)

    def _is_paddlex_backend(self) -> bool:
        return self._backend.startswith("paddlex")
    def _init_ocr(self, use_gpu: bool = False):
        import os
        import time
        import logging

        # 如果请求 GPU，先检查 CUDA 是否真的可用：
        # - PaddleX ONNX Runtime 内部硬性要求 CUDAExecutionProvider
        # - PaddleOCR transformers / paddle_static 也需要 CUDA 环境
        # - OCR 后端均不支持 DirectML，因此只有 CUDA 可用时才保留 GPU 请求
        if use_gpu:
            try:
                import onnxruntime as ort
                available = set(ort.get_available_providers())
                if "CUDAExecutionProvider" not in available:
                    print(
                        f"[OCR] 当前 OCR 后端需要 CUDA，"
                        f"可用 providers={available}，降级到 CPU"
                    )
                    use_gpu = False
            except Exception:
                use_gpu = False

        # 在导入 paddle 前禁用 oneDNN 与新 PIR API，规避 PaddlePaddle 3.x 已知崩溃
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("PADDLE_DISABLE_PIR", "1")
        # 提升 Paddle 相关日志级别，减少初始化时的海量输出
        for _name in ("paddle", "ppocr", "paddlex", "PaddleOCR", "", "urllib3", "requests"):
            _logger = logging.getLogger(_name)
            _logger.setLevel(logging.WARNING)
            for _handler in _logger.handlers[:]:
                _handler.setLevel(logging.WARNING)

        # PaddleX ONNX Runtime 后端单独初始化
        if self._user_engine in ("paddlex_onnx", "paddlex", "onnxruntime"):
            self._init_paddlex_ocr(use_gpu=use_gpu)
            return

        # engine=None 时自动选择：优先尝试更快的 PaddleX ONNX Runtime，
        # 不可用则回退到 PaddleOCR transformers / paddle_static
        if self._user_engine is None and self._try_init_paddlex_ocr(use_gpu=use_gpu):
            return

        t0 = time.perf_counter()
        from paddleocr import PaddleOCR
        t1 = time.perf_counter()
        if self.debug:
            print(f"[DEBUG] 导入 paddleocr 耗时: {(t1 - t0) * 1000:.1f}ms")
        engines_to_try = []
        if self._user_engine is not None:
            engines_to_try.append(self._user_engine)
        else:
            # PaddleX 不可用时，优先尝试 transformers（mobile 模型体积小），
            # 失败后再回退 paddle_static
            engines_to_try.extend(["transformers", None])

        last_error = None
        for engine in engines_to_try:
            kwargs = dict(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="gpu:0" if use_gpu else "cpu",
            )
            if engine is not None:
                kwargs["engine"] = engine

            # transformers 后端支持切换 mobile/server 模型
            if engine == "transformers" and self._model_size == "mobile":
                kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
                kwargs["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"

            try:
                t2 = time.perf_counter()
                # 临时屏蔽 Paddle 初始化时的海量内部输出（进度条、彩色日志、C 层日志等）。
                # 在 QProcess 管道环境下，C 层输出不走 Python sys.stdout，必须同时做 fd 级重定向，
                # 否则 stderr 管道会被填满导致子进程阻塞（表现为 OCR 加载“卡住”）。
                import sys
                import tempfile
                devnull = open(os.devnull, "w")
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = devnull
                sys.stderr = devnull
                # fd 级重定向：Windows 下重定向到真实临时文件而非 NUL，避免 tqdm 在个别环境下的兼容问题
                _tmpout = tempfile.NamedTemporaryFile(delete=False)
                _tmpfd = _tmpout.fileno()
                _old_stdout_fd = os.dup(1)
                _old_stderr_fd = os.dup(2)
                os.dup2(_tmpfd, 1)
                os.dup2(_tmpfd, 2)
                try:
                    self._ocr = PaddleOCR(**kwargs)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                    devnull.close()
                    os.dup2(_old_stdout_fd, 1)
                    os.dup2(_old_stderr_fd, 2)
                    os.close(_old_stdout_fd)
                    os.close(_old_stderr_fd)
                    _tmpout.close()
                    try:
                        os.unlink(_tmpout.name)
                    except Exception:
                        pass
                t3 = time.perf_counter()
                self._backend = f"paddleocr_{engine}" if engine else "paddleocr_paddle"
                self.device = kwargs["device"]
                # 该标记用于前端识别 OCR 已就绪并自动最小化窗口，不在界面上显示
                print(f"[OCR] 初始化成功: {self._backend} (模型: {self._model_size}, 设备: {self.device})")
                if self.debug:
                    print(f"[DEBUG] PaddleOCR(engine={engine or 'paddle'}) 构造耗时: {(t3 - t2) * 1000:.1f}ms")
                return
            except Exception as e:
                last_error = e
                t3 = time.perf_counter()
                print(f"[OCR] 后端 {engine or 'paddle'} 初始化失败: {e}")
                if self.debug:
                    print(f"[DEBUG] PaddleOCR(engine={engine or 'paddle'}) 尝试失败耗时: {(t3 - t2) * 1000:.1f}ms")
                continue

        # 全部失败才抛异常
        raise RuntimeError(f"PaddleOCR 初始化失败（已尝试 {engines_to_try}）: {last_error}")

    def _try_init_paddlex_ocr(self, use_gpu: bool = False) -> bool:
        """尝试初始化 PaddleX ONNX Runtime OCR；失败时返回 False 而不抛异常。"""
        try:
            self._init_paddlex_ocr(use_gpu=use_gpu)
            return True
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] PaddleX ONNX Runtime OCR 自动初始化失败，将回退 PaddleOCR: {e}")
            return False

    def _init_paddlex_ocr(self, use_gpu: bool = False):
        """使用 PaddleX create_pipeline 初始化 ONNX Runtime OCR 后端。"""
        import time
        from pathlib import Path
        from paddlex import create_pipeline

        t0 = time.perf_counter()
        if self._model_size == "server":
            det_model_name = "PP-OCRv5_server_det"
            rec_model_name = "PP-OCRv5_server_rec"
        else:
            det_model_name = "PP-OCRv5_mobile_det"
            rec_model_name = "PP-OCRv5_mobile_rec"

        config = {
            "pipeline_name": "OCR",
            "text_type": "general",
            "use_doc_preprocessor": False,
            "use_textline_orientation": False,
            "SubModules": {
                "TextDetection": {
                    "module_name": "text_detection",
                    "model_name": det_model_name,
                    "model_dir": self._det_model_dir,
                    "limit_side_len": 64,
                    "limit_type": "min",
                    "max_side_limit": 8192,
                    "thresh": 0.3,
                    "box_thresh": 0.6,
                    "unclip_ratio": 1.5,
                },
                "TextRecognition": {
                    "module_name": "text_recognition",
                    "model_name": rec_model_name,
                    "model_dir": self._rec_model_dir,
                    "batch_size": 6,
                    "score_thresh": 0.0,
                },
            },
        }

        device = "gpu:0" if use_gpu else "cpu"
        try:
            self._ocr = create_pipeline(
                "OCR",
                config=config,
                device=device,
                engine="onnxruntime",
            )
            self._backend = "paddlex_onnx"
            self.device = device
            t1 = time.perf_counter()
            print(
                f"[OCR] 初始化成功: {self._backend} "
                f"(det={det_model_name}, rec={rec_model_name}, "
                f"device={device}, det_dir={self._det_model_dir}, rec_dir={self._rec_model_dir})"
            )
            if self.debug:
                print(f"[DEBUG] PaddleX OCR 构造耗时: {(t1 - t0) * 1000:.1f}ms")
        except Exception as e:
            raise RuntimeError(f"PaddleX ONNX Runtime OCR 初始化失败: {e}") from e

    def recognize(self, image: np.ndarray, min_confidence: float = 0.6) -> list:
        if self._is_paddlex_backend():
            return self._recognize_paddlex(image, min_confidence)

        raw_result = self._ocr.predict(image)
        lines = []
        for res in raw_result:
            data = res.json if hasattr(res, "json") else {}
            if not data or "res" not in data:
                continue
            res_data = data["res"]
            # 必须用 rec_polys（与 rec_texts 一一对应），
            # dt_polys 包含未识别的误检框，会导致索引错位
            polys = res_data.get("rec_polys", [])
            texts = res_data.get("rec_texts", [])
            scores = res_data.get("rec_scores", [])
            for i in range(len(texts)):
                bbox = polys[i] if i < len(polys) else []
                text = texts[i]
                conf = float(scores[i]) if i < len(scores) else 0.0
                # 过滤过低置信度的噪声，避免误识别干扰匹配
                if bbox and text and conf >= min_confidence:
                    lines.append([bbox, (text, conf)])
        if self.debug:
            self._save_debug(image, lines, raw_result)
        return lines

    def _recognize_paddlex(self, image: np.ndarray, min_confidence: float = 0.6) -> list:
        """PaddleX ONNX Runtime 后端的结果解析，统一成 [bbox, (text, conf)] 格式。"""
        # PaddleX OCR 的 det/rec 模型默认期望 3 通道 BGR 输入；
        # 关键帧截图可能是 4 通道（BGRA），这里做转换并补齐小图尺寸。
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        min_h, min_w = 64, 64
        h, w = image.shape[:2]
        if h < min_h or w < min_w:
            pad_top = (min_h - h) // 2 if h < min_h else 0
            pad_bottom = (min_h - h) - pad_top if h < min_h else 0
            pad_left = (min_w - w) // 2 if w < min_w else 0
            pad_right = (min_w - w) - pad_left if w < min_w else 0
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )

        result = list(
            self._ocr.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        )[0]
        lines = []
        polys = result.get("rec_polys", [])
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        for poly, text, conf in zip(polys, texts, scores):
            if poly is None or not text:
                continue
            conf = float(conf)
            if conf >= min_confidence:
                lines.append([poly.tolist() if hasattr(poly, "tolist") else poly, (text, conf)])
        if self.debug:
            self._save_debug(image, lines, result)
        return lines

    def _save_debug(self, original: np.ndarray, lines: list, raw_result=None):
        """debug 模式下保存截图及 OCR 结果到 debug/ 目录。"""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        ts = int(time.time() * 1000)

        # 保存原始截图
        img_path = debug_dir / f"ocr_frame_{ts}.png"
        cv2.imencode(".png", original)[1].tofile(str(img_path))

        # 优先使用 PaddleOCR 官方 save_to_img 保存带框图，确保框坐标完全准确
        if raw_result is not None:
            try:
                official_img_path = str(debug_dir / f"ocr_frame_{ts}_official.png")
                for res in raw_result:
                    res.save_to_img(official_img_path)
                print(f"[DEBUG] 官方标注图已保存: {official_img_path}")
            except Exception as e:
                print(f"[DEBUG] 官方 save_to_img 失败，回退到自绘标注: {e}")
                annotated = self._annotate_image(original, lines)
                ann_path = debug_dir / f"ocr_frame_{ts}_annotated.png"
                cv2.imencode(".png", annotated)[1].tofile(str(ann_path))
                print(f"[DEBUG] 自绘标注图已保存: {ann_path}")

            # 使用官方 save_to_json 保存结构化结果
            try:
                official_json_path = str(debug_dir / f"ocr_frame_{ts}_official.json")
                for res in raw_result:
                    res.save_to_json(official_json_path)
                print(f"[DEBUG] 官方 JSON 已保存: {official_json_path}")
            except Exception as e:
                print(f"[DEBUG] 官方 save_to_json 失败: {e}")

        # 保存文本结果
        texts = []
        for line in lines:
            if not line:
                continue
            _, (text, conf) = line
            texts.append(f"{text}({conf:.2f})")
        txt_path = debug_dir / f"ocr_frame_{ts}.txt"
        txt_path.write_text("\n".join(texts), encoding="utf-8")
        print(f"[DEBUG] 截图已保存: {img_path}")
        print(f"[DEBUG] 识别到 {len(texts)} 行文本（已过滤置信度<0.3）")
        for t in texts:
            print(f"[DEBUG]   - {t}")

    def _annotate_image(self, image: np.ndarray, lines: list) -> np.ndarray:
        """在截图上绘制 OCR 识别框及文字（自绘回退方案）。"""
        from PIL import Image, ImageDraw, ImageFont
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 尝试加载 Windows 中文字体
        font = None
        for fp in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ]:
            try:
                font = ImageFont.truetype(fp, 20)
                break
            except Exception:
                pass
        if font is None:
            font = ImageFont.load_default()

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        for i, line in enumerate(lines):
            if not line:
                continue
            bbox, (text, conf) = line
            color = colors[i % len(colors)]
            pts = [(int(p[0]), int(p[1])) for p in bbox]
            # 绘制四边形框
            draw.polygon(pts, outline=color, width=2)
            # 在框左上角绘制文字
            x = min(p[0] for p in bbox)
            y = min(p[1] for p in bbox)
            label = f"{text} ({conf:.2f})"
            # 简单背景条避免文字被画面内容遮挡
            bbox_text = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            draw.rectangle([x, y - th - 4, x + tw, y], fill=(0, 0, 0))
            draw.text((x, y - th - 4), label, fill=color, font=font)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """归一化文本：去空格、统一横线、转大写，提升关卡名匹配率。"""
        text = text.replace(" ", "").replace("　", "")
        text = text.replace("—", "-").replace("–", "-").replace("—", "-")
        return text.upper()

    @staticmethod
    def _alnum_only(text: str) -> str:
        """仅保留字母和数字，用于模糊匹配。"""
        import re
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def _match_text(self, text: str, target: str) -> bool:
        """多级文本匹配：精确归一化 -> 仅字母数字模糊匹配。"""
        norm_text = self._normalize_text(text)
        norm_target = self._normalize_text(target)
        if norm_target in norm_text:
            return True
        # 回退：去掉所有符号后再比较（如 TD-7 vs TD7）
        return self._alnum_only(target) in self._alnum_only(text)

    def find_text(self, image: np.ndarray, target: str, lines: Optional[list] = None) -> Optional[tuple]:
        if lines is None:
            lines = self.recognize(image)
        # 先单框匹配
        for line in lines:
            if not line:
                continue
            bbox, (text, conf) = line
            if self._match_text(text, target):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                cx = int(sum(xs) / len(xs))
                cy = int(sum(ys) / len(ys))
                return cx, cy, text, conf
        # 回退：拼接相邻框文本，按 x 坐标从左到右排序
        if lines:
            sorted_lines = sorted(
                [l for l in lines if l],
                key=lambda l: min(p[0] for p in l[0])
            )
            combined_text = "".join(l[1][0] for l in sorted_lines)
            if self._match_text(combined_text, target):
                all_xs = []
                all_ys = []
                for line in sorted_lines:
                    for p in line[0]:
                        all_xs.append(p[0])
                        all_ys.append(p[1])
                cx = int(sum(all_xs) / len(all_xs))
                cy = int(sum(all_ys) / len(all_ys))
                return cx, cy, combined_text, 1.0
        return None

    def extract_all_text(self, image: np.ndarray) -> str:
        lines = self.recognize(image)
        texts = []
        for line in lines:
            if not line:
                continue
            _, (text, _) = line
            texts.append(text)
        return " ".join(texts)
