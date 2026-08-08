"""项目路径工具：统一处理开发环境与 PyInstaller 打包后的资源定位。"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """返回项目根目录。PyInstaller 打包后返回 exe 所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
RESOURCE_DIR = PROJECT_ROOT / "resource"

# 资源子目录
GAME_TEMPLATE_DIR = RESOURCE_DIR / "game_template"
GUI_TEMPLATE_DIR = RESOURCE_DIR / "gui_template"
GAME_DATA_DIR = RESOURCE_DIR / "game_data"
POSITION_DATA_DIR = GAME_DATA_DIR / "position_data"
SUMMONS_DIR = RESOURCE_DIR / "summons"
MODELS_DIR = RESOURCE_DIR / "models"


def game_template(name: str) -> Path:
    """返回游戏内模板图路径。"""
    return GAME_TEMPLATE_DIR / name


def gui_template(name: str) -> Path:
    """返回 GUI/界面模板图路径。"""
    return GUI_TEMPLATE_DIR / name


def game_data(name: str) -> Path:
    """返回游戏数据文件路径。"""
    return GAME_DATA_DIR / name


def position_data(name: str) -> Path:
    """返回位置/ROI 配置文件路径。"""
    return POSITION_DATA_DIR / name


def summon_config(name: str) -> Path:
    """返回召唤物 JSON 配置文件路径。"""
    return SUMMONS_DIR / name


def model(subdir: str, name: str) -> Path:
    """返回模型文件路径，subdir 如 ResNet/MoblienetV4/num/X_num_CNN/YOLO。"""
    return MODELS_DIR / subdir / name
