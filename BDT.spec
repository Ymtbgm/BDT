# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata, collect_data_files, collect_dynamic_libs

import os
import shutil
import site
import tempfile

# 项目根目录固定为 spec 文件所在目录，避免因为执行目录不同导致资源路径错误。
_project_root = os.path.abspath(SPECPATH)

# 获取当前 Python 环境的 site-packages 路径（兼容 conda/venv），避免写死 .venv。
_site_packages = None
for _sp in site.getsitepackages():
    if _sp.endswith('site-packages'):
        _site_packages = _sp
        break
if _site_packages is None:
    _site_packages = os.path.join(os.path.dirname(site.getusersitepackages()), 'site-packages')


# PyInstaller 在 Windows 上处理含中文路径的图标文件时容易报 FileNotFoundError，
# 先把图标复制到临时 ASCII 路径再使用。
_icon_src = os.path.join(_project_root, 'resource', 'gui_template', 'Icon.ico')
_icon_path = _icon_src
if os.path.exists(_icon_src):
    try:
        _icon_temp = os.path.join(tempfile.gettempdir(), 'BDT_Icon.ico')
        shutil.copy(_icon_src, _icon_temp)
        _icon_path = _icon_temp
    except Exception:
        _icon_path = _icon_src

datas = []
binaries = []

# 将 resource/ 下的静态资源（模板图、图标、levels.json、模型配置等）打包进输出目录
_resource_dir = os.path.join(_project_root, 'resource')
if os.path.isdir(_resource_dir):
    for _root, _dirs, _files in os.walk(_resource_dir):
        for _f in _files:
            _src = os.path.join(_root, _f)
            _dst = os.path.relpath(_root, _project_root)
            datas.append((_src, _dst))

hiddenimports = ['paddleocr', 'paddlex', 'paddle', 'cv2', 'numpy', 'pydantic', 'action', 'onnxruntime', 'requests', 'urllib3']
# 收集本地包下的所有子模块，防止 PyInstaller 静态分析遗漏
# 显式列出模块名更可靠（collect_submodules 在某些环境下可能无法解析本地包）
hiddenimports += [
    'gui', 'gui.app', 'gui.main_window', 'gui.widgets', 'gui.widgets.toast', 'gui.workers', 'gui.workers.levels_update_worker',
    'core', 'core.capture.capture', 'core.vision.ocr_engine', 'core.control.executor', 'core.map.grid_mapper',
    'core.special_behaviors', 'core.special_behaviors.base', 'core.special_behaviors.config_field',
    'core.special_behaviors.registry', 'core.special_behaviors.probability_checkpoint',
    'core.game_state.timer', 'core.game_state.operator_pool', 'core.game_state.ui_scale_check', 'core.vision.leak_detector', 'core.control.stage_selector',
    'core.control.retry_handler', 'core.game_state.cost_bar_calibration', 'core.game_state.cost_bar_start',
    'core.game_state.cost_bar_sync', 'core.game_state.cost_bar_sync_cc', 'core.game_state.region_state_timer',
    'core.game_state.summon_registry', 'core.vision.avatar_matcher', 'core.map.tile_pos',
    'core.vision.cost_recognition', 'core.recording.resolver', 'core.vision.digit_recognizer',
    'core.vision.skill_click_detector',
    'core.base.onnx_utils', 'core.base.logging_utils', 'core.vision.yolo_detector', 'core.base.paths',
    'core.update', 'core.update.levels_updater',
    'models', 'models.script_schema', 'models.raw_recording',
    'action',
]

# Cython 的数据文件（如 Utility/CppSupport.cpp）在异常 traceback 渲染时会被读取，
# 缺失会导致 paddle 后端初始化报错
datas += collect_data_files('Cython')

tmp_ret = collect_all('PyQt6.QtCore')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PyQt6.QtGui')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PyQt6.QtWidgets')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# PyQt6 平台插件有时无法被自动收集，手动补齐
_pyqt6_plugins = os.path.join(_site_packages, 'PyQt6', 'Qt6', 'plugins', 'platforms')
if os.path.exists(_pyqt6_plugins):
    datas += [(os.path.join(_pyqt6_plugins, 'qwindows.dll'), 'PyQt6/Qt6/plugins/platforms')]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mss')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ONNX Runtime 会被 digit_recognizer 动态加载，其执行提供程序 DLL 容易被 PyInstaller 遗漏
binaries += collect_dynamic_libs('onnxruntime')
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# paddleocr 依赖动态导入，必须用 collect_all 确保子模块和数据文件都被收集
# 注意：collect_all('paddle') 已知会导致 segfault，故仅收集 paddleocr
tmp_ret = collect_all('paddleocr')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# paddlex 与 modelscope 是 paddleocr 的硬依赖，且包含 YAML pipeline 配置等数据文件，
# 必须用 collect_all 收集，否则运行时会出现 "The pipeline (OCR) does not exist!"
tmp_ret = collect_all('paddlex')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('modelscope')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torchvision')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# ultralytics 会被 YOLO 检测器动态导入，必须完整收集其数据文件和子模块
#（否则打包后 YOLO 模型加载失败，数量检测会回退到慢得多的 OCR 路径）
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# PaddleOCR transformers 后端会动态加载 pp_ocrv5 系列自定义模型（含 lazy import 的子模块），
# 必须把四个变体全部完整收集，否则会出现 ImageProcessor 缺失错误。
for _pp_model in [
    'transformers.models.pp_ocrv5_mobile_det',
    'transformers.models.pp_ocrv5_mobile_rec',
    'transformers.models.pp_ocrv5_server_det',
    'transformers.models.pp_ocrv5_server_rec',
]:
    tmp_ret = collect_all(_pp_model)
    datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# paddle 的 C++ 扩展与 DLL 在 PyInstaller 静态分析中极易被遗漏，
# 导致 paddle 后端初始化报缺少 libpaddle.pyd 或 paddle/libs 下 DLL。
binaries += collect_dynamic_libs('paddle')
_paddle_base = os.path.join(_site_packages, 'paddle', 'base')
if os.path.exists(os.path.join(_paddle_base, 'libpaddle.pyd')):
    binaries += [(os.path.join(_paddle_base, 'libpaddle.pyd'), 'paddle/base')]
# libpaddle.pyd 额外依赖 common.dll 和 mkldnn.dll，PyInstaller 有时解析不到，手动补齐
_paddle_libs = os.path.join(_site_packages, 'paddle', 'libs')
for _dll_name in ('common.dll', 'mkldnn.dll'):
    _dll_path = os.path.join(_paddle_libs, _dll_name)
    if os.path.exists(_dll_path):
        binaries += [(_dll_path, 'paddle/libs')]

# imagesize 代码在 PyInstaller 分析中被遗漏，但其 dist-info 被 paddlex 的 ocr-core extra 检查
tmp_ret = collect_all('imagesize')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# paddlex 通过 importlib.metadata 检查依赖是否安装，缺失 dist-info 会导致
# is_dep_available() 返回 False，进而触发 DependencyError。
# 此处手动补齐 ocr-core extra 以及 transformers/paddle 引擎所需的 dist-info。
for _pkg in ['imagesize', 'pyclipper', 'pypdfium2', 'python_bidi', 'shapely', 'transformers', 'paddlepaddle']:
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass

a = Analysis(
    ['entry.py'],
    pathex=[_project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy', 'matplotlib', 'skimage', 'sklearn', 'PyQt5'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ArknightsAuto',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    icon=_icon_path,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ArknightsAuto',
)

# 将用户可替换的资源/脚本从 _internal 复制到 exe 同级目录，
# 使打包后的路径结构与开发环境一致，避免运行时找不到资源闪退。
_dist_dir = os.path.join(_project_root, 'dist', 'ArknightsAuto')
if 'DISTPATH' in globals():
    _dist_dir = os.path.join(DISTPATH, 'ArknightsAuto')
elif 'distpath' in globals():
    _dist_dir = os.path.join(distpath, 'ArknightsAuto')

import shutil
_resource_src = os.path.join(_dist_dir, '_internal', 'resource')
_resource_dst = os.path.join(_dist_dir, 'resource')
if os.path.isdir(_resource_src):
    shutil.copytree(_resource_src, _resource_dst, dirs_exist_ok=True)

_scripts_src = os.path.join(_project_root, 'scripts')
_scripts_dst = os.path.join(_dist_dir, 'scripts')
if os.path.isdir(_scripts_src):
    shutil.copytree(_scripts_src, _scripts_dst, dirs_exist_ok=True)

_example_src = os.path.join(_project_root, 'example_script.json')
_example_dst = os.path.join(_dist_dir, 'example_script.json')
if os.path.isfile(_example_src):
    shutil.copy2(_example_src, _example_dst)
