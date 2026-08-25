# 项目文件地图（map.md）

本文档按目录梳理每个 `.py` 文件的大致职责及其核心类/函数，供快速定位已有能力、避免重复实现。目录结构以项目根为基准。

---

## 1. 根级入口与脚本

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `entry.py` | 程序启动入口：根据参数决定启动 GUI 还是命令行后端 | `main()` |
| `main.py` | 后端命令行入口，组装 Runner、解析参数、启动脚本执行 | `Runner`, `main()` |
| `action.py` | 封装游戏输入操作与划火柴全局热键 | `configure_keys()`, `configure_matchstick()`, `start_matchstick_listener()`, `stop_matchstick_listener()`, `pause()`, `select_at()`, `select_operator_matchstick()`, `deploy_at()`, `retreat_at()`, `skill_at()`, `p_and_esc_click()`, `p_and_left_click()` |
| `calibrate.py` | 标定工具入口（费用条、数量 ROI 等） | `main()` |
| `check_alignment.py` | 检查窗口/地图对齐 | `main()` |
| `check_view.py` | 检查关卡 view 与屏幕投影对齐 | `main()` |
| `tmp_compare_models.py` | 临时：对比头像匹配模型性能 | `main()` |
| `tmp_ocr_test.py` | 临时：快速测试 OCR | `main()` |
| `tmp_run_resolver.py` | 临时：离线运行录制解析器 | `main()` |

---

## 2. `core/` — 核心逻辑

### 2.1 `core/base/` — 基础设施

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/base/constants.py` | 全局常量：ROI、阈值、键位、时间、校准参数 | 常量定义 |
| `core/base/paths.py` | 统一资源路径管理 | `get_project_root()`, `game_data()`, `game_template()`, `gui_template()`, `model()` |
| `core/base/logging_utils.py` | 统一日志工具 | `log_info()`, `log_error()`, `log_debug()`, `set_verbose()` |
| `core/base/onnx_utils.py` | ONNX Runtime 会话选项与执行 Provider | `create_session_options()`, `get_onnx_providers()` |

### 2.2 `core/capture/` — 截图

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/capture/capture.py` | 游戏窗口截图封装，支持 PrintWindow / mss / WGC 等后端 | `WindowCapture.capture()`, `WindowCapture.capture_roi()`, `WindowCapture.get_window_size()` |

### 2.3 `core/control/` — 控制流

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/control/executor.py` | **脚本执行器**：按时间轴调度 DEPLOY / RETREAT / SKILL / SPEED_UP / PAUSE / 特殊行为 | `ScriptExecutor.load_script()`, `ScriptExecutor.run()`, `ScriptExecutor.wait_until()`, `ScriptExecutor.set_cost_sync()`, `ScriptExecutor._execute_action()`, `ScriptExecutor._sync_to_frame()`, `ScriptExecutor._execute_batch()`, `ScriptExecutor._execute_cluster()` |
| `core/control/retry_handler.py` | 漏怪/失败/概率点失败后自动退出并重开关卡 | `StageRetryHandler.handle_leak_once()`, `StageRetryHandler._retry_stage()` |
| `core/control/stage_selector.py` | 自动选关、进入关卡、选择助战 | `StageSelector.enter_stage()`, `StageSelector._select_stage()`, `StageSelector._start_battle()` |

### 2.4 `core/game_state/` — 游戏状态与时序

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/game_state/region_state_timer.py` | **视觉计时器**：基于区域 B 倍率采样、费用条启动检测、暂停/倍率补偿 | `RegionStateTimer.start()`, `RegionStateTimer.stop()`, `RegionStateTimer.tick()`, `RegionStateTimer.get_elapsed_ms()`, `RegionStateTimer.adjust()`, `RegionStateTimer.shield_matchstick()` |
| `core/game_state/timer.py` | 基础计时器/时间推进工具 | 详见文件 |
| `core/game_state/cost_bar_calibration.py` | 费用条校准数据：普通 / 危机合约 tag | `CostBarCalibration`, `get_calibration()`, `list_calibrations()` |
| `core/game_state/cost_bar_start.py` | 通过费用条从 0 开始增长的帧判定游戏启动时刻 | `CostBarStartDetector.tick()`, `CostBarStartDetector.state` |
| `core/game_state/cost_bar_sync.py` | 费用条自然回复周期同步（普通模式） | `CostBarSync.current_frame()`, `CostBarSync.target_frame_index()` |
| `core/game_state/cost_bar_sync_cc.py` | 危机合约/多阶段费用条同步，支持按时间切换校准表 | `CostBarSyncCC.current_frame()`, `CostBarSyncCC.get_calibration()` |
| `core/game_state/operator_pool.py` | 跟踪场上干员/道具/召唤物与部署栏槽位 | `OperatorPool.deploy()`, `OperatorPool.retreat()`, `OperatorPool.use_skill()`, `OperatorPool.get_bar_index_pos()` |
| `core/game_state/summon_registry.py` | 召唤物与召唤者绑定关系 | `SummonRegistry.register()`, `SummonRegistry.get_summons_for_operator()` |
| `core/game_state/ui_scale_check.py` | 启动前检查 Windows UI 缩放是否为 100% | `check_ui_scale()` |

### 2.5 `core/map/` — 地图投影

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/map/grid_mapper.py` | 地图格子与屏幕像素双向映射，支持精确投影与等分回退 | `GridMapper.grid_to_pixel()`, `GridMapper.pixel_to_grid()`, `GridMapper.get_side_deploy_offset_vector()` |
| `core/map/tile_pos.py` | 基于 3D 透视投影计算格子屏幕坐标，从 `levels.json` 加载关卡数据 | `TilePosCalculator.get_screen_pos()`, `TilePosCalculator.hit_test()`, `load_stage_dimensions()`, `load_stage_tiles()` |

### 2.6 `core/recording/` — 录制与解析

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/recording/recorder.py` | **操作录制器**：监听鼠标/键盘，生成 RawRecording，调用解析器 | `ActionRecorder.start()`, `ActionRecorder.stop()`, `ActionRecorder.take_over()`, `ActionRecorder._on_click()`, `ActionRecorder._capture_squad_keyframes()`, `ActionRecorder._resolve_recording()` |
| `core/recording/resolver.py` | **离线解析器**：将 RawRecording 解析为 ScriptModel，识别干员/道具/召唤物/动作 | `OfflineResolver.resolve()`, `OfflineResolver._build_initial_bar_state()`, `OfflineResolver._process_actions_forward()`, `OfflineResolver._build_script()` |

### 2.7 `core/special_behaviors/` — 特殊行为（可扩展）

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/special_behaviors/__init__.py` | 导出注册表入口 | `get_registry()` |
| `core/special_behaviors/base.py` | 特殊行为抽象基类 | `SpecialBehavior.get_config_fields()`, `SpecialBehavior.execute()` |
| `core/special_behaviors/registry.py` | 自动扫描注册所有 `SpecialBehavior` 子类 | `SpecialBehaviorRegistry.discover()`, `get_registry()` |
| `core/special_behaviors/config_field.py` | 前端配置项描述 | `ConfigField` |
| `core/special_behaviors/probability_checkpoint.py` | 概率点检查入口行为 | `ProbabilityCheckpointBehavior.execute()` |
| `core/special_behaviors/probability_checkpoints/base.py` | 概率点具体检查方法基类 | `ProbabilityCheckpointMethod` |
| `core/special_behaviors/probability_checkpoints/registry.py` | 检查方法注册表 | `get_method_registry()` |
| `core/special_behaviors/probability_checkpoints/grid_target.py` | 格子目标存在性检查 | `GridTargetMethod.execute()` |
| `core/special_behaviors/probability_checkpoints/kill_count.py` | 击杀数检查 | `KillCountMethod.execute()` |

### 2.8 `core/update/` — 资源更新

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/update/levels_updater.py` | 从远端下载并更新 `levels.json` | `LevelsUpdater.check_update()`, `LevelsUpdater.download()`, `LevelsUpdater.apply_update()` |

### 2.9 `core/vision/` — 视觉识别

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `core/vision/ocr_engine.py` | OCR 引擎封装：PaddleOCR / PaddleX ONNX | `OCREngine.recognize()`, `OCREngine.find_text()`, `OCREngine.extract_all_text()` |
| `core/vision/cost_recognition.py` | 部署栏费用 OCR 与预处理 | `recognize_operator_costs()`, `recognize_costs_by_bar_index()`, `preprocess_cost_image()` |
| `core/vision/digit_recognizer.py` | ONNX 费用数字（0~99）与数量角标分类 | `DigitRecognizer.predict_cost()`, `DigitRecognizer.predict_quantity()` |
| `core/vision/avatar_matcher.py` | 干员头像匹配：模板 / ResNet / ONNX / LogoMiniCNN | `create_avatar_matcher()`, `AvatarMatcher.compute_score_matrix()` |
| `core/vision/yolo_detector.py` | YOLO 数量角标检测 | `QuantityBadgeDetector.detect()` |
| `core/vision/skill_click_detector.py` | YOLO 技能按钮可点击状态检测 | `SkillClickDetector.detect()`, `SkillClickDetector.is_clickable()` |
| `core/vision/leak_detector.py` | 漏怪/失败画面模板匹配检测 | `LeakDetector.check_once()`, `LeakDetector.start_monitoring()` |

---

## 3. `gui/` — 前端界面

### 3.1 主窗口与入口

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `gui/app.py` | GUI 应用入口：创建 QApplication、校验 UI 缩放 | `main()` |
| `gui/main_window.py` | 主窗口：组合各 Tab、配置持久化、共享状态 | `MainWindow._build_ui()`, `MainWindow._apply_config()`, `MainWindow._save_config()`, `MainWindow._apply_matchstick_config()`, `MainWindow.closeEvent()` |
| `gui/_window_effects.py` | Windows 窗口特效：去玻璃/圆角、置顶、工具窗口 | `remove_dwm_glass_border()`, `set_window_topmost()`, `set_tool_window_style()` |
| `gui/timer_overlay.py` | 半透明计时器悬浮窗 | `TimerOverlay.update_time()`, `TimerOverlay.set_pause_text()` |
| `gui/info_collection_overlay.py` | 录制提示/信息录入悬浮窗 | `InfoCollectionOverlay.set_phase()`, `InfoCollectionOverlay.set_time()`, `InfoCollectionOverlay.set_button_callbacks()` |

### 3.2 `gui/tabs/` — 各功能 Tab

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `gui/tabs/exec_tab.py` | 脚本执行 Tab：加载脚本、配置参数、启动后端进程 | `ExecTab._start_script()`, `ExecTab._stop_script()`, `ExecTab._on_stdout()` |
| `gui/tabs/editor_tab.py` | 脚本编辑 Tab：干员/道具/召唤物/动作时间轴编辑 | `EditorTab.load_script()`, `EditorTab.save_script()`, `EditorTab._refresh_action_table()` |
| `gui/tabs/timer_tab.py` | 计时器 Tab：启动/停止视觉计时器与悬浮窗 | `TimerTab._start_region_timer()`, `TimerTab._stop_region_timer()`, `TimerTab._on_timer_tick()` |
| `gui/tabs/recorder_tab.py` | 操作录制 Tab：录制参数、启动/停止录制、保存脚本 | `RecorderTab._start_recording()`, `RecorderTab._stop_recording()`, `RecorderTab._poll_recorder_state()` |
| `gui/tabs/matchstick_tab.py` | 划火柴 Tab：配置全局热键 | `MatchstickTab._build_ui()` |
| `gui/tabs/resource_tab.py` | 资源更新 Tab：`levels.json` 检查与下载 | `ResourceTab._on_check_update()`, `ResourceTab.trigger_auto_check()` |
| `gui/tabs/guide_tab.py` | 使用指南 Tab | `GuideTab._load_guide()` |

### 3.3 `gui/widgets/` 与 `gui/workers/`

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `gui/widgets/checked_combo_box.py` | 可多选勾选的下拉框 | `CheckedComboBox.set_checked_data()`, `CheckedComboBox.checked_data()` |
| `gui/widgets/toast.py` | 轻量提示弹窗 | `Toast.show_message()` |
| `gui/workers/levels_update_worker.py` | levels.json 更新后台线程 | `LevelsUpdateWorker.run()` |

---

## 4. `models/` — 数据模型

| 文件 | 主要职责 | 核心类/函数 |
|------|---------|------------|
| `models/script_schema.py` | 脚本数据模型与 JSON Schema | `ScriptModel`, `OperatorAction`, `ActionType`, `ItemInfo`, `SummonInfo`, `SummonBinding`, `ScriptModel.validate_deploy_directions()` |
| `models/raw_recording.py` | 原始录制数据模型 | `RawRecording`, `RawAction`, `Keyframe`, `KeyframeType` |

---

## 5. `tools/` — 调试/标定/分析工具

按用途分组：

### 标定与可视化
- `tools/calibrate_operator_cost_roi.py` — 标定干员费用 ROI
- `tools/calibrate_quantity_roi.py` — 标定数量角标 ROI
- `tools/calibrate_rate_templates.py` — 标定倍率检测模板
- `tools/calibrate_skill_retreat_roi.py` — 标定技能/撤退按钮 ROI
- `tools/visualize_deploy_bar_roi.py` — 可视化部署栏 ROI
- `tools/visualize_merged_strip.py` — 可视化合并数量条
- `tools/visualize_skill_retreat_roi.py` — 可视化技能/撤退按钮 ROI
- `tools/fit_skill_retreat_projection.py` — 拟合技能/撤退按钮世界锚点投影

### 地图与格子
- `tools/grid_position_debug.py` — 调试地图格子投影
- `tools/analyze_tile_offset.py` — 分析 tile 偏移
- `tools/tile_hit_debug.py` — 调试 tile 命中测试
- `tools/find_best_offset.py` — 寻找最佳模板匹配偏移

### 费用条与倍率
- `tools/analyze_cost_bar.py` — 分析费用条图像
- `tools/capture_cost_bar.py` — 捕获费用条样本
- `tools/capture_cost_bar_cc.py` — 捕获危机合约费用条样本
- `tools/rate_transition_diagnostic.py` — 诊断倍率切换
- `tools/region_timer_tool.py` — 命令行区域计时器

### OCR / 头像 / 数字
- `tools/capture_digit_dataset.py` — 捕获数字识别数据集
- `tools/compare_onnx_perf.py` — 对比 ONNX 模型性能
- `tools/benchmark_onnx_resnet.py` — ONNX ResNet 性能基准
- `tools/run_resolver.py` — 手动运行录制解析器

### 截图与 ROI
- `tools/benchmark_capture_backends.py` — 对比截图后端性能
- `tools/roi_capture_benchmark.py` — ROI 截图性能基准
- `tools/region_capture_verify.py` — 验证截图区域
- `tools/capture_region_templates.py` — 捕获区域模板
- `tools/capture_skill_state_dataset.py` — 捕获技能状态数据集
- `tools/skill_state_debug.py` — 调试技能可点击状态
- `tools/get_pixel.py` — 获取屏幕指定像素颜色
- `tools/debug_ui_scale.py` — 调试 UI 缩放检测
- `tools/wgc_pygame_benchmark.py` — WGC/Pygame 截图基准
- `tools/windows_capture_dynamic_benchmark.py` — Windows 动态截图基准
- `tools/windows_capture_window_hwnd_benchmark.py` — HWND 窗口截图基准

---

## 6. 核心数据流速查

```
录制流程：
  ActionRecorder（监听输入） -> RawRecording -> OfflineResolver -> ScriptModel

执行流程：
  ExecTab/Runner -> ScriptExecutor.load_script() -> ScriptExecutor.run()
  -> wait_until() -> _execute_action()/_execute_batch()/_execute_cluster()
  -> action.py 发送按键/鼠标 -> 游戏

计时流程：
  RegionStateTimer.start() -> tick() -> 费用条/倍率检测 -> _scaled_elapsed_ms
  -> TimerOverlay / InfoCollectionOverlay 显示

视觉依赖链：
  WindowCapture -> OCR / 模板匹配 / YOLO / 数字分类 -> OperatorPool / GridMapper
```

---

## 7. 新增能力前先查这里

| 想实现的能力 | 优先查看的文件 |
|-------------|--------------|
| 新增游戏输入/按键 | `action.py` |
| 新增脚本动作类型 | `models/script_schema.py`, `core/control/executor.py` |
| 新增特殊行为 | `core/special_behaviors/base.py`, `core/special_behaviors/registry.py`, `core/special_behaviors/probability_checkpoint.py` |
| 新增概率点检查方法 | `core/special_behaviors/probability_checkpoints/base.py`, `core/special_behaviors/probability_checkpoints/registry.py` |
| 新增视觉识别 | `core/vision/` 下已有 OCR/模板/YOLO/数字分类器 |
| 新增计时启动/同步策略 | `core/game_state/region_state_timer.py`, `core/game_state/cost_bar_sync_cc.py` |
| 新增 UI Tab / 控件 | `gui/tabs/`, `gui/widgets/` |
| 新增资源路径 | `core/base/paths.py` |
| 新增常量/阈值 | `core/base/constants.py` |
| 新增调试/标定工具 | `tools/` |

---

> 注：本地图为高层索引，具体函数签名与实现细节请直接阅读对应文件。若新增文件或核心函数，请同步更新本文件。
