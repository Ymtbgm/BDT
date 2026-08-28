"""诊断模型加载与推理后端信息，用于排查打包后变慢的问题。"""
import os
import sys
import time
from pathlib import Path

# 兼容打包环境
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).parent
else:
    ROOT = Path(__file__).resolve().parent.parent

print(f"project_root={ROOT}")
print(f"frozen={getattr(sys, 'frozen', False)}")

# 1. ONNX Runtime 可用 provider
print("\n--- ONNX Runtime ---")
try:
    import onnxruntime as ort
    print(f"version={ort.__version__}")
    print(f"available_providers={ort.get_available_providers()}")
except Exception as e:
    print(f"ERROR: {e}")

# 2. Torch 线程与设备
print("\n--- PyTorch ---")
try:
    import torch
    print(f"version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"default_num_threads={torch.get_num_threads()}")
    print(f"num_interop_threads={torch.get_num_interop_threads()}")
except Exception as e:
    print(f"ERROR: {e}")

# 3. 头像匹配器初始化与 warm-up
print("\n--- Avatar Matcher ---")
try:
    sys.path.insert(0, str(ROOT))
    from core.vision.avatar_matcher import create_avatar_matcher
    import numpy as np

    t0 = time.perf_counter()
    matcher = create_avatar_matcher(prefer_resnet=True, use_onnx=True, model_name="resnet18")
    t1 = time.perf_counter()
    print(f"init_time_ms={(t1 - t0) * 1000:.1f}")
    print(f"matcher_type={type(matcher).__name__}")
    if hasattr(matcher, "providers"):
        print(f"providers={matcher.providers}")

    # warm-up inference
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    _ = matcher.compute_score_matrix({"a": dummy}, [dummy])
    t1 = time.perf_counter()
    print(f"warmup_time_ms={(t1 - t0) * 1000:.1f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# 4. YOLO 检测器
print("\n--- YOLO Detector ---")
try:
    from core.vision.yolo_detector import QuantityBadgeDetector
    import numpy as np

    t0 = time.perf_counter()
    det = QuantityBadgeDetector()
    t1 = time.perf_counter()
    print(f"init_time_ms={(t1 - t0) * 1000:.1f}")
    print(f"available={det.available}")

    dummy = np.full((65, 2560, 3), 255, dtype=np.uint8)
    t0 = time.perf_counter()
    _ = det.detect(dummy)
    t1 = time.perf_counter()
    print(f"detect_time_ms={(t1 - t0) * 1000:.1f}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# 5. Digit Recognizer
print("\n--- Digit Recognizer ---")
try:
    from core.vision.digit_recognizer import DigitRecognizer

    t0 = time.perf_counter()
    rec = DigitRecognizer(use_gpu=False)
    t1 = time.perf_counter()
    print(f"init_time_ms={(t1 - t0) * 1000:.1f}")
    print(f"available={rec.available}")
    print(f"providers={rec._providers}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n--- done ---")
