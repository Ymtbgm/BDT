import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.avatar_matcher import ONNXResNetAvatarMatcher, _default_onnx_path


def main():
    input_size = 112
    model_path = _default_onnx_path(input_size)
    if not model_path.exists():
        print(f"ONNX 模型不存在: {model_path}，请先运行 skill_state_debug.py --resnet --onnx 导出")
        return

    matcher = ONNXResNetAvatarMatcher(model_path, input_size=input_size)

    # 构造一张和 75x75 ROI 差不多的测试图
    img = np.random.randint(50, 150, (75, 75, 3), dtype=np.uint8)
    cv2.circle(img, (37, 37), 22, (0, 0, 255), -1)

    # 预热
    matcher._extract_batch([img])
    matcher._extract_batch([img])

    iterations = 200
    total_times = []
    preprocess_times = []
    inference_times = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        tensor = matcher._preprocess(img)
        t1 = time.perf_counter()
        matcher.session.run(None, {matcher.input_name: np.expand_dims(tensor, axis=0)})
        t2 = time.perf_counter()

        preprocess_times.append((t1 - t0) * 1000)
        inference_times.append((t2 - t1) * 1000)
        total_times.append((t2 - t0) * 1000)

    print(f"ONNX ResNet18 {input_size}x{input_size} tight-loop benchmark ({iterations} iters)")
    print(f"  总平均:   {sum(total_times)/len(total_times):.3f} ms")
    print(f"  预处理:   {sum(preprocess_times)/len(preprocess_times):.3f} ms")
    print(f"  推理:     {sum(inference_times)/len(inference_times):.3f} ms")


if __name__ == "__main__":
    main()
