import os
import sys
import time
from pathlib import Path
import numpy as np

# 可在外部通过环境变量控制： OMP_NUM_THREADS=1 python tools/benchmark_onnx_threads.py
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.base.paths import model

p = model("ResNet", "resnet18_avatar_matcher_224.onnx")
print(f"model_path={p}")
print(f"exists={p.exists()}")
print(f"onnxruntime version={ort.__version__}")
print(f"providers={ort.get_available_providers()}")
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")

for intra in (1, 4, 8, 24):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = intra
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    # warm-up
    sess.run(None, {in_name: x})
    t0 = time.perf_counter()
    for _ in range(20):
        sess.run(None, {in_name: x})
    elapsed = (time.perf_counter() - t0) * 1000 / 20
    print(f"intra={intra}: avg={elapsed:.2f}ms")
