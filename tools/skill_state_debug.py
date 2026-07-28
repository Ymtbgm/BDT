import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.avatar_matcher import (
    ensure_resnet18_onnx,
    ONNXResNetAvatarMatcher,
    ResNetAvatarMatcher,
    LogoMiniCNNMatcher,
)
from core.capture import WindowCapture
from core.tile_pos import TilePosCalculator


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def load_image(path: Path) -> np.ndarray:
    """加载原始图像（保留颜色与透明通道）。"""
    if not path.exists():
        raise FileNotFoundError(f"模板不存在: {path}")
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"无法读取模板: {path}")
    return img


def load_template(path: Path) -> np.ndarray:
    """加载模板并转为灰度（用于 NCC）。"""
    return to_gray(load_image(path))


def ncc_score(roi: np.ndarray, template: np.ndarray) -> float:
    th, tw = template.shape[:2]
    if th > roi.shape[0] or tw > roi.shape[1]:
        return 0.0
    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    return float(np.max(result))


def resolve_template_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return Path(__file__).parent.parent / "core" / "resource" / raw


def load_stage_info(code: str | None, name: str | None) -> dict | None:
    """从 levels.json 查找关卡信息，返回包含 width/height/view 的字典。"""
    levels_path = Path(__file__).parent.parent / "core" / "resource" / "levels.json"
    if not levels_path.exists():
        return None
    try:
        levels = json.loads(levels_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for lv in levels:
        if code and lv.get("code") == code:
            return lv
        if name and lv.get("name") == name:
            return lv
    return None


def main():
    parser = argparse.ArgumentParser(description="测试头顶 Logo 状态模板的相似度与推理耗时")
    parser.add_argument("--stage-code", type=str, default=None, help="关卡 code（用于精确 view）")
    parser.add_argument("--stage-name", type=str, default=None, help="关卡 name")
    parser.add_argument("--rows", type=int, default=None, help="地图行数（默认从 levels.json 读取）")
    parser.add_argument("--cols", type=int, default=None, help="地图列数（默认从 levels.json 读取）")
    parser.add_argument("--row", type=int, required=True, help="干员所在格子行")
    parser.add_argument("--col", type=int, required=True, help="干员所在格子列")
    parser.add_argument("--active", type=str, required=True, help="弹药 logo 模板路径（相对 core/resource 或绝对路径）")
    parser.add_argument("--inactive", type=str, required=True, help="非技能 logo 模板路径（相对 core/resource 或绝对路径）")
    parser.add_argument("--roi-offset-x", type=int, default=-35)
    parser.add_argument("--roi-offset-y", type=int, default=-240)
    parser.add_argument("--roi-w", type=int, default=75)
    parser.add_argument("--roi-h", type=int, default=75)
    parser.add_argument("--resnet", action="store_true", help="使用 ResNet18（默认 ONNX Runtime，CPU）计算余弦相似度")
    parser.add_argument("--pytorch", action="store_true", help="--resnet 时强制使用 PyTorch 而非 ONNX")
    parser.add_argument("--mini-cnn", action="store_true", help="使用自训练 logo_mini_cnn.onnx 进行 active/inactive 二分类")
    parser.add_argument("--benchmark-iters", type=int, default=0, help="采集完成后对同一 ROI 再做 N 次纯推理 benchmark（0 表示不跑）")
    parser.add_argument("--input-size", type=int, default=112, help="ResNet 输入尺寸（默认 112）")
    parser.add_argument("--frames", type=int, default=30, help="采样帧数")
    parser.add_argument("--interval", type=float, default=0.05, help="帧间隔（秒），0 表示连续采集")
    args = parser.parse_args()

    # 若指定了关卡，从 levels.json 读取地图尺寸，否则和 grid_position_debug 不一致
    if args.stage_code or args.stage_name:
        stage_info = load_stage_info(args.stage_code, args.stage_name)
        if stage_info is not None:
            json_rows = stage_info.get("height")
            json_cols = stage_info.get("width")
            if args.rows is None and isinstance(json_rows, int):
                args.rows = json_rows
            if args.cols is None and isinstance(json_cols, int):
                args.cols = json_cols
            if args.rows is None or args.cols is None:
                print(f"警告：levels.json 中该关卡尺寸不完整（height={json_rows}, width={json_cols}），使用默认尺寸")
        else:
            print(f"警告：未在 levels.json 中找到关卡 code={args.stage_code} name={args.stage_name}，使用默认尺寸")

    # 仍未指定则使用默认 7x9
    rows = args.rows or 7
    cols = args.cols or 9
    print(f"使用地图尺寸: {rows}x{cols}")

    capture = WindowCapture()
    w, h = capture.get_window_size()
    print(f"窗口尺寸: {w}x{h}")

    tile_calc = TilePosCalculator(
        screen_width=w,
        screen_height=h,
        grid_rows=rows,
        grid_cols=cols,
        stage_code=args.stage_code,
        stage_name=args.stage_name,
    )

    scale = min(w / 2560, h / 1600)
    ox = int(args.roi_offset_x * scale)
    oy = int(args.roi_offset_y * scale)
    roi_w = max(1, int(args.roi_w * scale))
    roi_h = max(1, int(args.roi_h * scale))

    center_x, center_y = tile_calc.get_screen_pos(args.row, args.col, side=False)
    left = capture.monitor.get("left", 0)
    top = capture.monitor.get("top", 0)
    abs_x = left + center_x + ox
    abs_y = top + center_y + oy
    print(f"normal view 中心: ({center_x},{center_y})")
    print(f"ROI 绝对坐标: ({abs_x},{abs_y},{roi_w},{roi_h})  缩放: {scale:.3f}")

    active_color = load_image(resolve_template_path(args.active))
    inactive_color = load_image(resolve_template_path(args.inactive))
    active_template = to_gray(active_color)
    inactive_template = to_gray(inactive_color)
    print(f"active 模板: {active_color.shape}")
    print(f"inactive 模板: {inactive_color.shape}")

    mini_cnn_matcher = None
    resnet_matcher = None
    resnet_template_features = None
    if args.mini_cnn:
        mini_cnn_path = Path(__file__).parent.parent / "core" / "resource" / "models" / "logo_mini_cnn.onnx"
        if not mini_cnn_path.exists():
            raise FileNotFoundError(f"找不到 mini CNN 模型: {mini_cnn_path}")
        mini_cnn_matcher = LogoMiniCNNMatcher(mini_cnn_path)
        print(f"Mini CNN 已加载: {mini_cnn_path}")
    elif args.resnet:
        if args.pytorch:
            resnet_matcher = ResNetAvatarMatcher(input_size=args.input_size)
            print(f"PyTorch ResNet18 已加载，输入尺寸: {args.input_size}")
        else:
            onnx_path = ensure_resnet18_onnx(input_size=args.input_size)
            resnet_matcher = ONNXResNetAvatarMatcher(onnx_path, input_size=args.input_size)
            print(f"ONNX ResNet18 已加载: {onnx_path}，输入尺寸: {args.input_size}")

        names = ["active", "inactive"]
        imgs = [active_color, inactive_color]
        resnet_template_features = resnet_matcher._extract_batch(imgs)
        if resnet_template_features.ndim == 1:
            resnet_template_features = resnet_template_features.reshape(1, -1)
        resnet_template_features = resnet_template_features / (
            np.linalg.norm(resnet_template_features, axis=1, keepdims=True) + 1e-8
        )
        print(f"ResNet18 模板特征已预提取（使用原始彩图），形状: {resnet_template_features.shape}")

    ncc_active_times = []
    ncc_inactive_times = []
    mini_cnn_total_times = []
    mini_cnn_preprocess_times = []
    mini_cnn_inference_times = []
    resnet_total_times = []
    resnet_preprocess_times = []
    resnet_inference_times = []
    resnet_similarity_times = []

    debug_dir = Path("debug") / "skill_state_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 预热一次，避免把 ONNX 首次推理的初始化时间算进平均
    warmup_roi = capture.capture_roi(abs_x, abs_y, roi_w, roi_h)
    if mini_cnn_matcher is not None:
        _ = mini_cnn_matcher.predict(warmup_roi)
        print("Mini CNN 已预热")
    if resnet_matcher is not None:
        _ = resnet_matcher._extract_batch([warmup_roi])
        print("ResNet 已预热")

    for i in range(args.frames):
        roi = capture.capture_roi(abs_x, abs_y, roi_w, roi_h)
        roi_gray = to_gray(roi)

        t0 = time.perf_counter()
        active_score = ncc_score(roi_gray, active_template)
        ncc_active_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        inactive_score = ncc_score(roi_gray, inactive_template)
        ncc_inactive_times.append(time.perf_counter() - t0)

        winner = "active" if active_score > inactive_score else "inactive"
        print(f"[{i+1:02d}] NCC active={active_score:.3f} inactive={inactive_score:.3f} -> {winner}  "
              f"time_a={ncc_active_times[-1]*1000:.2f}ms time_i={ncc_inactive_times[-1]*1000:.2f}ms")

        resnet_scores = None
        if resnet_matcher is not None:
            t0 = time.perf_counter()
            if isinstance(resnet_matcher, ONNXResNetAvatarMatcher):
                # 对 ONNX 版本拆分为：预处理、推理、相似度
                t1 = time.perf_counter()
                tensor = resnet_matcher._preprocess(roi)
                t2 = time.perf_counter()
                raw = resnet_matcher.session.run(
                    None, {resnet_matcher.input_name: np.expand_dims(tensor, axis=0)}
                )[0]
                t3 = time.perf_counter()
                roi_feat = raw.reshape(1, -1)
                roi_feat = roi_feat / (np.linalg.norm(roi_feat, axis=1, keepdims=True) + 1e-8)
                sims = (resnet_template_features @ roi_feat.T).flatten()
                t4 = time.perf_counter()
                resnet_total_times.append((t4 - t0) * 1000)
                resnet_preprocess_times.append((t2 - t1) * 1000)
                resnet_inference_times.append((t3 - t2) * 1000)
                resnet_similarity_times.append((t4 - t3) * 1000)
            else:
                # PyTorch 版本只统计端到端
                roi_feat = resnet_matcher._extract_batch([roi])
                if roi_feat.ndim == 1:
                    roi_feat = roi_feat.reshape(1, -1)
                roi_feat = roi_feat / (np.linalg.norm(roi_feat, axis=1, keepdims=True) + 1e-8)
                sims = (resnet_template_features @ roi_feat.T).flatten()
                resnet_total_times.append((time.perf_counter() - t0) * 1000)
                resnet_preprocess_times.append(0.0)
                resnet_inference_times.append(resnet_total_times[-1])
                resnet_similarity_times.append(0.0)

            resnet_scores = {"active": float(sims[0]), "inactive": float(sims[1])}
            print(f"      ResNet active={resnet_scores['active']:.3f} inactive={resnet_scores['inactive']:.3f} "
                  f"total={resnet_total_times[-1]:.2f}ms "
                  f"pre={resnet_preprocess_times[-1]:.2f}ms "
                  f"infer={resnet_inference_times[-1]:.2f}ms "
                  f"sim={resnet_similarity_times[-1]:.2f}ms")

        mini_cnn_scores = None
        if mini_cnn_matcher is not None:
            t0 = time.perf_counter()
            tensor = mini_cnn_matcher._preprocess(roi)
            t1 = time.perf_counter()
            logits = mini_cnn_matcher.session.run(
                None, {mini_cnn_matcher.input_name: tensor}
            )[0]
            t2 = time.perf_counter()
            probs = mini_cnn_matcher._softmax(logits)
            t3 = time.perf_counter()
            mini_cnn_total_times.append((t3 - t0) * 1000)
            mini_cnn_preprocess_times.append((t1 - t0) * 1000)
            mini_cnn_inference_times.append((t2 - t1) * 1000)
            mini_cnn_scores = {
                "active": float(probs[0, 0]),
                "inactive": float(probs[0, 1]),
            }
            winner_cnn = "active" if mini_cnn_scores["active"] > mini_cnn_scores["inactive"] else "inactive"
            print(f"      MiniCNN A={mini_cnn_scores['active']:.3f} I={mini_cnn_scores['inactive']:.3f} -> {winner_cnn} "
                  f"total={mini_cnn_total_times[-1]:.2f}ms "
                  f"pre={mini_cnn_preprocess_times[-1]:.2f}ms "
                  f"infer={mini_cnn_inference_times[-1]:.2f}ms")

        # 保存最后一帧可视化
        if i == args.frames - 1:
            canvas = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            text1 = f"NCC A={active_score:.2f} I={inactive_score:.2f}"
            cv2.putText(canvas, text1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            if resnet_scores:
                text2 = f"RN A={resnet_scores['active']:.2f} I={resnet_scores['inactive']:.2f}"
                cv2.putText(canvas, text2, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            if mini_cnn_scores:
                text3 = f"CNN A={mini_cnn_scores['active']:.2f} I={mini_cnn_scores['inactive']:.2f}"
                cv2.putText(canvas, text3, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            ts = int(time.time() * 1000)
            roi_path = debug_dir / f"roi_{args.row}_{args.col}_{ts}.png"
            ok, encoded = cv2.imencode(".png", canvas)
            if ok:
                roi_path.write_bytes(encoded.tobytes())
                print(f"已保存 ROI: {roi_path}")

            # 同时保存一张带 ROI 矩形的全屏叠加图，便于和 grid_position_debug 对比
            full = capture.capture()
            if full.shape[2] == 4:
                full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)
            rx = center_x + ox
            ry = center_y + oy
            cv2.rectangle(full, (rx, ry), (rx + roi_w, ry + roi_h), (0, 255, 255), 2)
            cv2.drawMarker(full, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            cv2.putText(full, f"ROI ({rx},{ry}) {roi_w}x{roi_h}", (rx, max(0, ry - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            full_path = debug_dir / f"full_{args.row}_{args.col}_{ts}.png"
            ok, encoded = cv2.imencode(".png", full)
            if ok:
                full_path.write_bytes(encoded.tobytes())
                print(f"已保存全屏叠加图: {full_path}")

        if args.interval > 0:
            time.sleep(args.interval)

    print("\n平均耗时:")
    avg_a = sum(ncc_active_times) / len(ncc_active_times) * 1000
    avg_i = sum(ncc_inactive_times) / len(ncc_inactive_times) * 1000
    print(f"  NCC active:   {avg_a:.3f} ms")
    print(f"  NCC inactive: {avg_i:.3f} ms")
    print(f"  NCC total:    {avg_a+avg_i:.3f} ms")
    if mini_cnn_total_times:
        print(f"  MiniCNN total: {sum(mini_cnn_total_times)/len(mini_cnn_total_times):.3f} ms")
        print(f"    预处理:      {sum(mini_cnn_preprocess_times)/len(mini_cnn_preprocess_times):.3f} ms")
        print(f"    推理:        {sum(mini_cnn_inference_times)/len(mini_cnn_inference_times):.3f} ms")
    if resnet_total_times:
        print(f"  ResNet total: {sum(resnet_total_times)/len(resnet_total_times):.3f} ms")
        if resnet_preprocess_times and any(resnet_preprocess_times):
            print(f"    预处理:     {sum(resnet_preprocess_times)/len(resnet_preprocess_times):.3f} ms")
            print(f"    推理:       {sum(resnet_inference_times)/len(resnet_inference_times):.3f} ms")
            print(f"    相似度:     {sum(resnet_similarity_times)/len(resnet_similarity_times):.3f} ms")

    if args.benchmark_iters > 0:
        print(f"\ntight-loop benchmark: 对最后一帧 ROI 连续推理 {args.benchmark_iters} 次")
        if mini_cnn_matcher is not None:
            bench_preprocess = []
            bench_inference = []
            bench_total = []
            for _ in range(args.benchmark_iters):
                t0 = time.perf_counter()
                tensor = mini_cnn_matcher._preprocess(roi)
                t1 = time.perf_counter()
                _ = mini_cnn_matcher.session.run(None, {mini_cnn_matcher.input_name: tensor})
                t2 = time.perf_counter()
                bench_preprocess.append((t1 - t0) * 1000)
                bench_inference.append((t2 - t1) * 1000)
                bench_total.append((t2 - t0) * 1000)
            print(f"  [MiniCNN] 总平均: {sum(bench_total)/len(bench_total):.3f} ms  预处理: {sum(bench_preprocess)/len(bench_preprocess):.3f} ms  推理: {sum(bench_inference)/len(bench_inference):.3f} ms")
        if isinstance(resnet_matcher, ONNXResNetAvatarMatcher):
            bench_preprocess = []
            bench_inference = []
            bench_total = []
            for _ in range(args.benchmark_iters):
                t0 = time.perf_counter()
                tensor = resnet_matcher._preprocess(roi)
                t1 = time.perf_counter()
                _ = resnet_matcher.session.run(
                    None, {resnet_matcher.input_name: np.expand_dims(tensor, axis=0)}
                )[0]
                t2 = time.perf_counter()
                bench_preprocess.append((t1 - t0) * 1000)
                bench_inference.append((t2 - t1) * 1000)
                bench_total.append((t2 - t0) * 1000)
            print(f"  [ResNet]  总平均: {sum(bench_total)/len(bench_total):.3f} ms  预处理: {sum(bench_preprocess)/len(bench_preprocess):.3f} ms  推理: {sum(bench_inference)/len(bench_inference):.3f} ms")


if __name__ == "__main__":
    main()
