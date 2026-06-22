# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata, collect_data_files, collect_dynamic_libs

datas = []
binaries = []
import os
_project_root = os.getcwd()

hiddenimports = ['paddleocr', 'paddlex', 'paddle', 'cv2', 'numpy', 'pydantic', 'action']
# 收集本地包下的所有子模块，防止 PyInstaller 静态分析遗漏
# 显式列出模块名更可靠（collect_submodules 在某些环境下可能无法解析本地包）
hiddenimports += [
    'gui', 'gui.app', 'gui.main_window',
    'core', 'core.capture', 'core.ocr_engine', 'core.executor', 'core.grid_mapper',
    'core.timer', 'core.operator_pool', 'core.leak_detector', 'core.stage_selector',
    'core.retry_handler',
    'models', 'models.script_schema',
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
_pyqt6_plugins = os.path.join(_project_root, '.venv', 'Lib', 'site-packages', 'PyQt6', 'Qt6', 'plugins', 'platforms')
if os.path.exists(_pyqt6_plugins):
    datas += [(os.path.join(_pyqt6_plugins, 'qwindows.dll'), 'PyQt6/Qt6/plugins/platforms')]
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mss')
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
_paddle_base = os.path.join(_project_root, '.venv', 'Lib', 'site-packages', 'paddle', 'base')
if os.path.exists(os.path.join(_paddle_base, 'libpaddle.pyd')):
    binaries += [(os.path.join(_paddle_base, 'libpaddle.pyd'), 'paddle/base')]
# libpaddle.pyd 额外依赖 common.dll 和 mkldnn.dll，PyInstaller 有时解析不到，手动补齐
_paddle_libs = os.path.join(_project_root, '.venv', 'Lib', 'site-packages', 'paddle', 'libs')
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
    icon=os.path.join(_project_root, 'core', 'resource', 'Icon.ico'),
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


