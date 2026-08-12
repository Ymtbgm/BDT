"""根据标定数据拟合技能和撤退按钮的等效世界锚点。

使用 debug/calibrate_skill_retreat_roi/<timestamp>/positions.json 中的实测中心，
结合对应关卡的 view_side，通过 side view 透视投影反求按钮在地图世界坐标系中的
固定锚点 P，使得 projected(P) 最接近实测屏幕中心。

拟合完成后会输出可直接写入 core/recording/recorder.py 的锚点常量。
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.map.tile_pos import TilePosCalculator, load_stage_dimensions

# 标定目录 timestamp -> 关卡代号
# 注：7-18 / 7-17 / 11-5 的 view 极为接近，互相交换对拟合结果影响很小。
CALIBRATION_MAPPING = {
    "1786507028253": "7-18",
    "1786507098600": "TO-9",
    "1786507176882": "TO-6",
    "1786508379216": "TO-4",
    "1786508445228": "TO-1",
    "1786508567598": "11-5",
    "1786510042676": "0-1",
    "1786510212912": "7-12",
    "1786510314204": "13-10",
    "1786510394090": "14-10",
    "1786533687536": "TO-EX-5",
}


def load_observations():
    """返回 skill / retreat 各自的观测列表：[(TilePosCalculator, sx, sy), ...]。"""
    base = ROOT / "debug" / "calibrate_skill_retreat_roi"
    skill, retreat = [], []
    for ts, stage in CALIBRATION_MAPPING.items():
        path = base / ts / "positions.json"
        if not path.exists():
            print(f"[跳过] 找不到标定文件: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        dims = load_stage_dimensions(stage)
        if dims is None:
            print(f"[跳过] 找不到关卡尺寸: {stage}")
            continue
        cols, rows = dims
        calc = TilePosCalculator(2560, 1600, rows, cols, stage_code=stage)
        sc = data["skill"]["base_center"]
        rc = data["retreat"]["base_center"]
        skill.append((calc, sc["x"], sc["y"]))
        retreat.append((calc, rc["x"], rc["y"]))
    return skill, retreat


def residuals(P, observations):
    px, py, pz = P
    res = []
    for calc, sx, sy in observations:
        matrix = calc._get_transform_matrix(side=True)
        cx, cy, _, cw = np.dot(matrix, np.array([px, py, pz, 1.0]))
        psx = (1 + cx / cw) / 2 * calc.screen_width
        psy = (1 - cy / cw) / 2 * calc.screen_height
        res.append(psx - sx)
        res.append(psy - sy)
    return np.array(res)


def fit_anchor(observations, init=(0.0, 0.0, 0.0), max_iter=200):
    P = np.array(init, dtype=float)
    prev_err = float("inf")
    for _ in range(max_iter):
        r = residuals(P, observations)
        err = float(np.sum(r ** 2))
        if err >= prev_err - 1e-9:
            break
        prev_err = err
        eps = 1e-5
        J = np.zeros((len(r), 3))
        for j in range(3):
            dP = P.copy()
            dP[j] += eps
            J[:, j] = (residuals(dP, observations) - r) / eps
        try:
            delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        except Exception:
            break
        best_err = err
        best_P = P
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            cand = P + alpha * delta
            cand_err = float(np.sum(residuals(cand, observations) ** 2))
            if cand_err < best_err:
                best_err = cand_err
                best_P = cand
        if np.allclose(best_P, P, atol=1e-9):
            break
        P = best_P
    return P, prev_err


def report(name, observations, P):
    print(f"\n{name} anchor: ({P[0]:.6f}, {P[1]:.6f}, {P[2]:.6f})")
    total_err = 0.0
    for calc, sx, sy in observations:
        matrix = calc._get_transform_matrix(side=True)
        cx, cy, _, cw = np.dot(matrix, np.array([*P, 1.0]))
        psx = (1 + cx / cw) / 2 * calc.screen_width
        psy = (1 - cy / cw) / 2 * calc.screen_height
        err = math.hypot(psx - sx, psy - sy)
        total_err += err ** 2
        print(f"  pred=({psx:7.1f}, {psy:7.1f})  meas=({sx:7.1f}, {sy:7.1f})  err={err:5.1f}px")
    rmse = math.sqrt(total_err / len(observations))
    print(f"  RMSE = {rmse:.2f}px")


def main():
    skill_obs, retreat_obs = load_observations()
    if not skill_obs or not retreat_obs:
        print("没有足够的标定数据")
        return

    best_skill, best_retreat = None, None
    best_skill_err, best_retreat_err = float("inf"), float("inf")
    inits = [
        (0, 0, 0),
        (0, 5, 0),
        (0, 0, 5),
        (5, 0, 0),
        (0, -5, 0),
        (0, 0, -5),
        (-5, 0, 0),
    ]
    for init in inits:
        P, err = fit_anchor(skill_obs, init=init)
        if err < best_skill_err:
            best_skill_err = err
            best_skill = P
        P, err = fit_anchor(retreat_obs, init=init)
        if err < best_retreat_err:
            best_retreat_err = err
            best_retreat = P

    report("SKILL", skill_obs, best_skill)
    report("RETREAT", retreat_obs, best_retreat)

    print("\n可直接写入 core/recording/recorder.py 的常量：")
    print(f"    _SKILL_ANCHOR   = ({best_skill[0]:.6f}, {best_skill[1]:.6f}, {best_skill[2]:.6f})")
    print(f"    _RETREAT_ANCHOR = ({best_retreat[0]:.6f}, {best_retreat[1]:.6f}, {best_retreat[2]:.6f})")


if __name__ == "__main__":
    main()
