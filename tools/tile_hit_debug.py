"""Side 视角格子命中实时调试工具。

用法：
    python tools/tile_hit_debug.py --stage-code 14-7
    python tools/tile_hit_debug.py --stage-code AS-9 --no-side
    python tools/tile_hit_debug.py --rows 8 --cols 12

启动后会弹出一个置顶小窗口，实时显示鼠标当前位置、窗口相对坐标、
hit_test 命中的格子（row, col）、该格子的 heightType / buildableType / tileKey，
以及鼠标到格子中心的距离。

若 hit_test 未命中任何格子，会回退到“最近中心”并标注 [nearest]。

按 F9 可记录当前状态，记录会追加保存到 debug/tile_hit_debug/ 目录下的
stage-code_时间戳.txt（或 manual_时间戳.txt）。
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tkinter as tk
from pynput.mouse import Controller
from pynput import keyboard

from core.capture.capture import WindowCapture
from core.map.tile_pos import TilePosCalculator, load_stage_dimensions


class TileHitDebugTool:
    def __init__(
        self,
        stage_code: str | None = None,
        rows: int | None = None,
        cols: int | None = None,
        side: bool = True,
    ):
        self.mouse = Controller()
        self.capture = WindowCapture()
        self.side = side
        self.stage_code = stage_code

        w, h = self.capture.get_window_size()

        if stage_code:
            dims = load_stage_dimensions(stage_code)
            if dims:
                grid_cols, grid_rows = dims
            else:
                print(f"警告：未找到 {stage_code} 的尺寸，使用手动/默认值")
                grid_rows = rows or 7
                grid_cols = cols or 9
        else:
            grid_rows = rows or 7
            grid_cols = cols or 9

        self.calc = TilePosCalculator(
            w, h, grid_rows, grid_cols, stage_code=stage_code
        )

        self.records: list[dict] = []
        self._last_record_time = 0.0
        self._record_cooldown = 0.3  # F9 防抖，避免一次按键触发多次

        out_dir = Path("debug") / "tile_hit_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = stage_code or "manual"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = out_dir / f"{prefix}_{ts}.txt"

        self.root = tk.Tk()
        self.root.title(f"Tile Hit Debug - {'side' if side else 'normal'}")
        self.root.attributes("-topmost", True)
        self.root.geometry("320x200+100+100")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        labels = [
            "mouse_screen",
            "mouse_window",
            "grid",
            "tile",
            "distance",
            "recorded",
        ]
        self.labels = {}
        for key in labels:
            lbl = tk.Label(
                self.root,
                text=f"{key}: ...",
                anchor="w",
                justify="left",
                font=("Consolas", 10),
            )
            lbl.pack(fill="x", padx=8, pady=2)
            self.labels[key] = lbl

        self.labels["recorded"].config(text="按 F9 记录 | 已记录: 0")

        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()

        self.update()
        self.root.mainloop()

    def _on_key_press(self, key):
        try:
            if key == keyboard.Key.f9:
                now = time.time()
                if now - self._last_record_time > self._record_cooldown:
                    self._last_record_time = now
                    self.root.after(0, self._record_current)
        except Exception:
            pass

    def _record_current(self):
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "stage_code": self.stage_code,
            "side": self.side,
            "screen": list(self._current_screen),
            "window": list(self._current_window),
            "grid": self._current_grid,
            "mode": self._current_mode,
            "distance": round(self._current_dist, 2) if self._current_dist else None,
            "tile": self._current_tile,
        }
        self.records.append(record)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.labels["recorded"].config(
            text=f"按 F9 记录 | 已记录: {len(self.records)} -> {self.log_path.name}"
        )
        print(f"[F9 记录] {record}")

    def _on_close(self):
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        print(f"\n共记录 {len(self.records)} 条，保存至: {self.log_path}")
        self.root.destroy()

    def _find_nearest_grid(self, win_x: int, win_y: int):
        best = None
        best_dist = float("inf")
        for r in range(self.calc.grid_rows):
            for c in range(self.calc.grid_cols):
                cx, cy = self.calc.get_screen_pos(r, c, side=self.side)
                d = (cx - win_x) ** 2 + (cy - win_y) ** 2
                if d < best_dist:
                    best_dist = d
                    best = (r, c)
        return best, best_dist ** 0.5 if best else None

    def update(self):
        screen_x, screen_y = self.mouse.position
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        win_x = screen_x - left
        win_y = screen_y - top

        self._current_screen = (screen_x, screen_y)
        self._current_window = (win_x, win_y)

        grid = self.calc.hit_test(win_x, win_y, side=self.side)
        mode = "hit_test"
        if grid is None:
            grid, dist = self._find_nearest_grid(win_x, win_y)
            mode = "nearest"
        else:
            cx, cy = self.calc.get_screen_pos(*grid, side=self.side)
            dist = ((cx - win_x) ** 2 + (cy - win_y) ** 2) ** 0.5

        self._current_grid = grid
        self._current_mode = mode
        self._current_dist = dist

        tile_text = "-"
        tile_dict = None
        if grid and self.calc.tiles:
            r, c = grid
            try:
                tile = self.calc.tiles[r][c]
                ht = tile.get("heightType", 0)
                bt = tile.get("buildableType", 0)
                tk_key = tile.get("tileKey", "unknown")
                tile_text = f"ht={ht} bt={bt} key={tk_key}"
                tile_dict = {"heightType": ht, "buildableType": bt, "tileKey": tk_key}
            except Exception:
                pass
        self._current_tile = tile_dict

        self.labels["mouse_screen"].config(
            text=f"screen: ({screen_x}, {screen_y})"
        )
        self.labels["mouse_window"].config(
            text=f"window: ({win_x}, {win_y})"
        )
        self.labels["grid"].config(
            text=f"grid: {grid} [{mode}]"
        )
        self.labels["tile"].config(text=f"tile: {tile_text}")
        self.labels["distance"].config(
            text=f"dist: {dist:.1f}px" if dist is not None else "dist: -"
        )

        self.root.after(50, self.update)


def main():
    parser = argparse.ArgumentParser(
        description="Side 视角格子命中实时调试工具"
    )
    parser.add_argument(
        "--stage-code", type=str, help="关卡 code，自动从 levels.json 读取尺寸和 view"
    )
    parser.add_argument("--rows", type=int, help="手动指定地图行数")
    parser.add_argument("--cols", type=int, help="手动指定地图列数")
    parser.add_argument(
        "--no-side", action="store_true", help="使用普通视角而非 side 视角"
    )
    args = parser.parse_args()

    TileHitDebugTool(
        stage_code=args.stage_code,
        rows=args.rows,
        cols=args.cols,
        side=not args.no_side,
    )


if __name__ == "__main__":
    main()
