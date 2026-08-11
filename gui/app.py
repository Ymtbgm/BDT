import json
import os
import sys
from pathlib import Path

# 避免 Qt 重复设置 DPI awareness 导致 Windows "拒绝访问" 警告。
# 若进程已被其他库设置为某种 DPI awareness，Qt 的默认设置会失败；
# 这里通过关闭 qt.qpa.window 日志来抑制该提示。
os.environ.setdefault("QT_QPA_PLATFORM", "windows:dpiawareness=0")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

from PyQt6.QtWidgets import QApplication
from core.base.paths import get_project_root
from core.game_state.ui_scale_check import check_ui_scale
from gui.main_window import MainWindow


def _config_path() -> Path:
    return get_project_root() / "config.json"


def _load_config() -> dict:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(config: dict) -> None:
    path = _config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    app = QApplication(sys.argv)
    config = _load_config()
    if not check_ui_scale(config=config):
        sys.exit(0)
    if config.get("ui_scale_check_disabled"):
        _save_config(config)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
