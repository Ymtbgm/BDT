import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess_simple(img_bgr, size):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (size, size))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
    tensor = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(tensor, axis=0).astype(np.float32)


def make_session(path, threads):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def benchmark_session(session, input_name, size, iters=300):
    img = np.random.randint(50, 150, (75, 75, 3), dtype=np.uint8)
    cv2.circle(img, (37, 37), 22, (0, 0, 255), -1)
    # warm
    for _ in range(5):
        session.run(None, {input_name: preprocess_simple(img, size)})
    times = []
    for _ in range(iters):
        x = preprocess_simple(img, size)
        t0 = time.perf_counter()
        session.run(None, {input_name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return sum(times) / len(times)


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    models = [
        ("ResNet_test full", base / "ResNet_test" / "resnet18_112x112.onnx"),
        ("our full temp", base / "core" / "resource" / "models" / "resnet18_112_full_temp.onnx"),
        ("our trimmed", base / "core" / "resource" / "models" / "resnet18_avatar_matcher_112.onnx"),
    ]
    threads_options = [None, 4, 8]
    for threads in threads_options:
        label = f"threads={threads}" if threads else "default threads"
        print(f"\n=== {label} ===")
        for name, path in models:
            sess = make_session(path, threads)
            input_name = sess.get_inputs()[0].name
            avg = benchmark_session(sess, input_name, 112)
            print(f"  {name}: {avg:.3f} ms")
