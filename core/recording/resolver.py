import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from core.vision.avatar_matcher import AvatarMatcherBase, create_avatar_matcher
from core.vision.ocr_engine import OCREngine
from core.vision.digit_recognizer import DigitRecognizer
from core.vision.yolo_detector import QuantityBadgeDetector
from core.vision import cost_recognition
import core.base.constants as constants
from core.base.paths import position_data
from models.raw_recording import RawRecording, RawAction, KeyframeType
from models.script_schema import ScriptModel, OperatorAction, ActionType, ItemInfo, SummonInfo, SummonBinding


class _DeployInfo:
    """单个 DEPLOY 动作在预分类阶段的结果。"""

    raw: RawAction
    bar_index: int
    name: str
    score: float
    is_item: bool
    is_unknown: bool
    best_total: int
    true_total: Optional[int]
    best_crop: Optional[np.ndarray]
    cost: Optional[int]
    quantity: Optional[int]


@dataclass
class _AddSummonInference:
    """记录一次 ADD_SUMMON 推断，便于反推完成后做 OCR 修正。"""

    target_name: str
    action_time_ms: int
    screenshot_time_ms: int
    screenshot_bar_index: int
    screenshot_total: int
    predicted_diff: int


@dataclass
class _InitialSlotInfo:
    """初始部署区（TEAM_BAR）某个 slot 的识别结果。"""

    bar_index: int
    name: Optional[str] = None
    cost: Optional[int] = None
    quantity: Optional[int] = None
    is_item: bool = False
    is_unknown: bool = False
    score: float = 0.0


@dataclass
class _InfiniteItem:
    """无限使用初始道具（无数量角标，可反复返回部署栏）。"""

    original_bar_index: int
    expected_cost: int
    name: str
    present: bool = True
    avatar: Optional[np.ndarray] = None


@dataclass
class _SlotState:
    """某一时刻部署栏某个 slot 的状态（从右往左 0 开始）。"""

    name: str
    is_item: bool = False
    is_summon: bool = False
    is_infinite: bool = False
    original_bar_index: Optional[int] = None
    quantity: Optional[int] = None
    cost: Optional[int] = None
    avatar: Optional[np.ndarray] = None


@dataclass
class _SummonObs:
    """一次召唤物 slot 观察记录，解析阶段用临时 obs_id 标识，最后统一聚类。"""

    obs_id: str
    avatars: List[np.ndarray] = field(default_factory=list)
    costs: List[int] = field(default_factory=list)
    quantities: List[int] = field(default_factory=list)
    first_seen_time_ms: int = 0
    first_seen_bar_index: int = 0


class OfflineResolver:
    """将 RawRecording（关键帧 + 占位操作）离线解析为 ScriptModel。

    解析逻辑（基于 pre-deploy 整栏图）：
      1. 用编队界面关键帧建立“名称 -> 头像模板”库；
      2. 从初始 TEAM_BAR 建立第一张 bar_slots 状态；
      3. 每次 DEPLOY 的 bar 关键帧为 mouseDown 时的 pre-deploy 状态：
         - 全宽数量条 OCR，检测自上次动作以来是否有新召唤物/道具；
         - 根据拖拽时的 bar_index 换算到实际总槽位数下的真实 slot 索引；
         - 直接取出该 slot 作为本次部署对象；
         - 模拟部署后状态（干员移除 / 召唤物或道具数量减 1）；
      4. RETREAT / SKILL 根据场上格子反查；
      5. 输出 ScriptModel。
    """

    # 与 core/recorder.py 和 core/constants.py 保持一致，用于从整栏关键帧中
    # 恢复窗口尺寸并裁剪出单个 slot 头像/费用/数量。
    # 注意：recorder 在截取整栏时把 top 上移了 20px（y = 1390 - 20），因此
    # 解析器里 bar_img 顶部对应的窗口 y 是 1370 而不是 1390。
    # 数量 ROI 位于屏幕最底部，因此整栏高度需要覆盖到窗口底部（1370 -> 1600）。
    _BAR_CAPTURE_TOP_RATIO = 1360 / 1600
    _BAR_CAPTURE_HEIGHT_RATIO = 240 / 1600
    _BAR_AVATAR_SIZE_RATIO = 120 / 1600
    _BAR_AVATAR_Y_OFFSET_RATIO = -10 / 120
    _BAR_CENTER_Y_RATIO = 1500 / 1600

    # active_slot 自身位置相对于普通 target 的垂直形变（像素，基于 1600h）
    _ACTIVE_SELF_Y_SHIFT_PX = 25

    # 数量 OCR ROI：覆盖部署栏底部 1535~1600 区域，确保拖拽开始时也能拍到数量角标
    _QUANTITY_ROI_Y_RATIO = 1535 / 1600
    _QUANTITY_ROI_H_RATIO = 65 / 1600

    # 数量角标最小尺寸（以 1600h 为基准），过小则视为 OCR 误识别噪点
    # 正常数量角标约 40x30，取 50% 作为下限
    _QUANTITY_MIN_W_PX_1600 = 20
    _QUANTITY_MIN_H_PX_1600 = 15

    # 数量角标最大尺寸（以 1600h 为基准），过大则视为 UI 元素/误检
    # 正常数量角标宽度不超过 60，高度不超过 40
    _QUANTITY_MAX_W_PX_1600 = 80
    _QUANTITY_MAX_H_PX_1600 = 60

    # 召唤物模板匹配阈值：对未知/召唤物模板要求更高置信度，避免同一召唤物被多次创建不同占位
    _SUMMON_MATCH_THRESHOLD = 0.80

    # 召唤物最终聚类阈值：最后对 obs_id 做全局聚类时，相似度超过该阈值才视为同一召唤物
    _SUMMON_CLUSTER_THRESHOLD = 0.85

    # 无限道具在场检测置信度阈值：低于该值的结果不用于切换在场状态
    _INFINITE_ITEM_PRESENCE_CONF_THRESHOLD = 0.6

    # 冷却召唤物红像素检测参数（基于 HSV，对暗红/酒红背景更鲁棒）
    _RED_COOLDOWN_HUE_LOW = 10
    _RED_COOLDOWN_HUE_HIGH = 170
    _RED_COOLDOWN_SAT_MIN = 30
    _RED_COOLDOWN_VAL_MIN = 30
    _RED_COOLDOWN_MIN_RATIO = 0.40

    def __init__(
        self,
        raw: RawRecording,
        session_dir: Optional[Path] = None,
        avatar_matcher: Optional[AvatarMatcherBase] = None,
        ocr: Optional[OCREngine] = None,
        digit_recognizer: Optional[DigitRecognizer] = None,
        match_threshold: float = 0.70,
        debug: bool = False,
        avatar_model_name: str = "resnet18",
        log_callback: Optional[Callable[[str], None]] = None,
        initial_deployed: Optional[Dict[Tuple[int, int], str]] = None,
        initial_bar_state: Optional[List[Dict]] = None,
    ):
        self.raw = raw
        self.debug = debug
        self._log_callback = log_callback
        self._match_threshold = match_threshold
        self._matcher = avatar_matcher
        self.ocr = ocr
        self._digit_recognizer = digit_recognizer or DigitRecognizer(use_gpu=True)
        self._yolo_detector = QuantityBadgeDetector()
        self._avatar_model_name = avatar_model_name
        self._session_dir = session_dir or Path("debug") / "recordings" / raw.session_id

        self._templates: Dict[str, np.ndarray] = {}
        self._unknown_counter = 0
        self._quantity_strip_debug_counter = 0
        self._quantity_slot_debug_counter = 0
        self._operators: List[str] = []
        self._items: List[ItemInfo] = []
        self._summons: List[SummonInfo] = []
        self._actions: List[OperatorAction] = []
        self._deployed: Dict[Tuple[int, int], str] = dict(initial_deployed) if initial_deployed else {}
        self._initial_bar_state: Optional[List[Dict]] = initial_bar_state

        self._item_count_hint = int(raw.hints.get("initial_item_count", 0))
        self._support_count = int(raw.hints.get("support_count", 0))
        self._item_bar_index: Dict[str, int] = {}
        self._infinite_items: Dict[int, _InfiniteItem] = {}
        self._debug_log_path = self._session_dir / "resolve.log"

        # 正向 bar_slots 状态机使用
        self._prev_bar_state: List[_SlotState] = []
        self._initial_operator_order: List[str] = []
        self._operator_index: Dict[str, int] = {}
        self._operator_info: Dict[str, _SlotState] = {}
        self._item_initial_quantity: Dict[str, int] = {}
        self._remaining_item_count = self.raw.initial_item_count

        # 两阶段解析使用（保留字段兼容，但新逻辑主要用 _prev_bar_state）
        self._deploy_infos: Dict[int, _DeployInfo] = {}
        self._item_usage_counts: Dict[str, int] = {}
        self._moved_add_summon_times: Set[int] = set()
        self._summon_deploy_counts: Dict[str, int] = {}
        self._summon_costs: Dict[str, int] = {}
        self._add_summon_inferences: List[_AddSummonInference] = []
        self._initial_summon_quantities: Dict[str, int] = {}
        self._initial_slot_info: Dict[int, _InitialSlotInfo] = {}

        # 召唤物 obs_id 方案：解析阶段每个召唤物 slot 都是独立观察，最后聚类合并
        self._summon_obs: Dict[str, _SummonObs] = {}
        self._summon_obs_counter: int = 0
        self._obs_id_to_final: Dict[str, str] = {}

        # 保存每次 DEPLOY 前后的 bar 状态快照，供后续检测消失召唤物使用
        self._deploy_snapshots: List[Tuple[int, List[_SlotState], List[_SlotState], Optional[str]]] = []

        # 保存每个 DEPLOY 关联的名称卡关键帧，按占位名称分组，用于离线 OCR 重命名
        self._name_card_kf_by_name: Dict[str, List[str]] = {}

        # 缓存：避免重复加载关键帧图片和重复 OCR
        self._image_cache: Dict[str, np.ndarray] = {}
        self._quantity_boxes_cache: Dict[int, List[Tuple[List, float]]] = {}
        self._ocr_cost_cache: Dict[Tuple[int, int, int, int], Optional[int]] = {}

        # 数量 ROI 标定配置缓存 (total_slots, calibrations dict)
        self._quantity_roi_total: Optional[int] = None
        self._quantity_roi_calibrations: Optional[Dict[str, List[dict]]] = None

        # operator_cost logo ROI 标定配置缓存（头像/费用 X 位置）
        self._operator_cost_roi_total: Optional[int] = None
        self._operator_cost_roi_calibrations: Optional[Dict[str, List[dict]]] = None

    def _log(self, message: str):
        """在 debug 模式时同时输出到控制台、写入会话日志文件，并可选回调到前端。"""
        if not self.debug:
            return
        line = f"[解析器] {message}"
        print(line)
        if self._log_callback is not None:
            try:
                self._log_callback(line)
            except Exception:
                pass
        if self._debug_log_path is not None:
            try:
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _support_operator_names(self) -> Set[str]:
        """识别助战干员：优先使用初始部署区头像解析出的顺序。

        初始 bar 状态是按从右到左解析的，因此最右侧的干员排在最前面；
        转为从左到右后，助战干员（通常在干员区域最右）位于末尾。
        若初始顺序不可用，则回退到名称“助战”前缀或原列表末尾假设。
        """
        if self._support_count <= 0:
            return set()
        if self._initial_operator_order:
            return set(self._initial_operator_order[-self._support_count :])
        support_names = [n for n in self._operators if n.startswith("助战")]
        if len(support_names) < self._support_count:
            fallback = [
                n
                for n in self._operators[-self._support_count :]
                if n not in support_names
            ]
            support_names.extend(
                fallback[: self._support_count - len(support_names)]
            )
        return set(support_names[: self._support_count])

    def resolve(self) -> ScriptModel:
        self._log(
            f"开始解析 RawRecording: session={self.raw.session_id}, "
            f"actions={len(self.raw.actions)}, ocr={'可用' if self.ocr else '不可用'}"
        )
        if self.ocr is not None:
            self._log(
                f"OCR 后端: {self.ocr._backend}, device: {getattr(self.ocr, 'device', 'unknown')}"
            )
        if self._matcher is not None:
            providers = getattr(self._matcher, "providers", None)
            if providers:
                self._log(f"头像匹配器 providers: {providers}")
        total_t0 = time.perf_counter()

        def _step(name: str, fn, *args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            self._log(f"[耗时] {name}: {(time.perf_counter() - t0) * 1000:.1f}ms")
            return result

        _step("加载编队模板", self._load_squad_templates)
        if self._initial_bar_state is not None:
            _step("应用脚本执行后的部署区状态", self._apply_initial_bar_state)
        else:
            _step("构建初始部署区状态", self._build_initial_bar_state)
        _step("记录初始召唤物数量", self._record_initial_summon_quantities)
        _step("正向处理所有动作", self._process_actions_forward)
        self._obs_id_to_final = _step("召唤物聚类", self._cluster_summon_obs)
        _step("重写召唤物名称", self._rewrite_actions_with_final_summon_names)
        _step("检测移除的召唤物", self._detect_removed_summons)
        _step("OCR 重命名名称卡", self._rewrite_deploy_card_names)
        relabeled = _step("识别冷却召唤物", self._identify_cooldown_summons)
        _step("清理冗余召唤物动作", self._cleanup_redundant_summon_actions, relabeled)
        _step("按数量修剪冗余 ADD_SUMMON", self._prune_redundant_add_summons_by_quantity)
        _step("修正 ADD_SUMMON delta", self._correct_add_summon_deltas)
        script = _step("构建最终脚本", self._build_script)

        self._log(
            f"解析完成: total={(time.perf_counter() - total_t0) * 1000:.1f}ms, "
            f"operators={len(script.operators)}, items={len(script.items)}, "
            f"summons={len(script.summons)}, actions={len(script.actions)}"
        )
        return script

    # ------------------------------------------------------------------
    # 数量 ROI 标定配置加载
    # ------------------------------------------------------------------
    def _load_quantity_roi_config(self, total_slots: int) -> Optional[Dict[str, List[dict]]]:
        """加载并缓存指定 total_slots 的数量 ROI 标定配置。"""
        if (
            self._quantity_roi_total == total_slots
            and self._quantity_roi_calibrations is not None
        ):
            return self._quantity_roi_calibrations

        candidates = [
            position_data(f"quantity_roi_config_total{total_slots}.json"),
            position_data("quantity_roi_config.json"),
        ]
        calibrations = None
        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_cals = data.get("calibrations", {})
                # 标定工具按 total_slots 分组保存；优先取当前 total_slots 分组
                calibrations = raw_cals.get(str(total_slots))
                # 兼容旧格式：顶层直接是 active_slot -> list
                if calibrations is None and raw_cals and not isinstance(next(iter(raw_cals.values())), dict):
                    calibrations = raw_cals
                if calibrations:
                    self._log(f"已加载数量 ROI 标定配置: {path.name} (total={total_slots})")
                    break
            except Exception as e:
                self._log(f"加载数量 ROI 配置失败 {path}: {e}")

        self._quantity_roi_total = total_slots
        self._quantity_roi_calibrations = calibrations
        return calibrations

    def _get_calibrated_quantity_roi(
        self,
        total_slots: int,
        active_slot: int,
        target_slot: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """返回标定的数量 ROI：(cx_ratio, cy_ratio, half_w_ratio, half_h_ratio)。

        total_slots <= 12 或找不到标定时返回 None，调用方应回退到动态 ROI。
        """
        if total_slots <= 12:
            return None
        calibrations = self._load_quantity_roi_config(total_slots)
        if not calibrations:
            return None
        active_key = str(active_slot)
        if active_key not in calibrations:
            return None
        for roi in calibrations[active_key]:
            if roi.get("slot") == target_slot:
                # 中心从 x1/x2/y1/y2 计算，保留原有的 +0.005 下偏修正；
                # 这样用户手动改 y1/y2 才会真正移动 ROI 中心，而不是只改变高度。
                cx = (roi["x1_ratio"] + roi["x2_ratio"]) / 2
                cy = (roi["y1_ratio"] + roi["y2_ratio"]) / 2 + 0.005
                half_w = (roi["x2_ratio"] - roi["x1_ratio"]) / 2
                half_h = (roi["y2_ratio"] - roi["y1_ratio"]) / 2
                return cx, cy, half_w, half_h
        return None

    def _get_quantity_roi_xranges(
        self,
        total_slots: int,
        active_slot: Optional[int] = None,
    ) -> Optional[Dict[int, Tuple[float, float]]]:
        """返回每个 slot 的数量角标在整栏图片中的标定 x 范围（像素）。

        如果指定了 active_slot 且标定中存在，则使用该 active_slot 的标定；
        否则取所有 active_slot 标定的并集，以兼容不确定 active_slot 的场景。
        total_slots <= 12 或找不到标定时返回 None。
        """
        if total_slots <= 12:
            return None
        calibrations = self._load_quantity_roi_config(total_slots)
        if not calibrations:
            return None

        active_key = str(active_slot) if active_slot is not None else None
        if active_key is not None and active_key in calibrations:
            roi_list = calibrations[active_key]
        else:
            # 合并所有 active_slot 的 x 范围
            merged: Dict[int, List[Tuple[float, float]]] = {}
            for roi_list in calibrations.values():
                for roi in roi_list:
                    slot = roi.get("slot")
                    if slot is None:
                        continue
                    merged.setdefault(slot, []).append(
                        (roi["x1_ratio"], roi["x2_ratio"])
                    )
            result: Dict[int, Tuple[float, float]] = {}
            for slot, ranges in merged.items():
                x1 = min(r[0] for r in ranges)
                x2 = max(r[1] for r in ranges)
                result[slot] = (x1, x2)
            return result

        result: Dict[int, Tuple[float, float]] = {}
        for roi in roi_list:
            slot = roi.get("slot")
            if slot is None:
                continue
            result[slot] = (roi["x1_ratio"], roi["x2_ratio"])
        return result

    def _load_operator_cost_roi_config(self, total_slots: int) -> Optional[Dict[str, List[dict]]]:
        """加载并缓存 operator_cost logo 的 X 位置标定配置。"""
        if (
            self._operator_cost_roi_total == total_slots
            and self._operator_cost_roi_calibrations is not None
        ):
            return self._operator_cost_roi_calibrations

        candidates = [
            position_data(f"operator_cost_roi_config_total{total_slots}.json"),
            position_data("operator_cost_roi_config.json"),
        ]
        calibrations = None
        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_cals = data.get("calibrations", {})
                calibrations = raw_cals.get(str(total_slots))
                # 兼容旧格式：顶层直接是 active_slot -> list
                if calibrations is None and raw_cals and not isinstance(next(iter(raw_cals.values())), dict):
                    calibrations = raw_cals
                if calibrations:
                    self._log(f"已加载 operator_cost ROI 标定配置: {path.name} (total={total_slots})")
                    break
            except Exception as e:
                self._log(f"加载 operator_cost ROI 配置失败 {path}: {e}")

        self._operator_cost_roi_total = total_slots
        self._operator_cost_roi_calibrations = calibrations
        return calibrations

    def _get_calibrated_operator_cost_roi(
        self,
        total_slots: int,
        active_slot: int,
        target_slot: int,
    ) -> Optional[Tuple[float, float, float, float]]:
        """返回标定的 operator_cost logo ROI：(cx_ratio, cy_ratio, half_w_ratio, half_h_ratio)。

        total_slots <= 12 或找不到标定时返回 None。
        """
        if total_slots <= 12:
            return None
        calibrations = self._load_operator_cost_roi_config(total_slots)
        if not calibrations:
            return None

        rois = None
        if isinstance(calibrations, dict):
            active_key = str(active_slot)
            rois = calibrations.get(active_key)
            # 未按 active_slot 分组时，把整个 dict 的 values 拼起来查找
            if rois is None:
                rois = [roi for sublist in calibrations.values() if isinstance(sublist, list) for roi in sublist]
        elif isinstance(calibrations, list):
            rois = calibrations

        if not rois:
            return None
        for roi in rois:
            if roi.get("slot") == target_slot:
                cx = roi["cx_ratio"]
                cy = roi["cy_ratio"]
                half_w = (roi["x2_ratio"] - roi["x1_ratio"]) / 2
                half_h = (roi["y2_ratio"] - roi["y1_ratio"]) / 2
                return cx, cy, half_w, half_h
        return None

    # ------------------------------------------------------------------
    # 模板库加载
    # ------------------------------------------------------------------
    def _load_squad_templates(self):
        """从编队关键帧加载头像模板与名称。"""
        squad_names: List[str] = list(self.raw.hints.get("squad_names", []))
        avatar_keyframes = [
            k for k in self.raw.keyframes.values()
            if k.type == KeyframeType.SQUAD_AVATAR
        ]
        avatar_keyframes.sort(key=lambda k: k.bar_index or 0)

        for kf in avatar_keyframes:
            idx = kf.bar_index or 0
            name = squad_names[idx] if idx < len(squad_names) else f"__squad_{idx}__"
            img = self._load_keyframe_image(kf.id)
            if img is None:
                continue
            self._templates[name] = img
            if name not in self._operators:
                self._operators.append(name)
            self._log(f"加载编队模板: {name}")

        if self._templates:
            if self._matcher is not None:
                self._log(
                    f"使用录制器预热的头像匹配器，跳过模板特征预计算: count={len(self._templates)}"
                )
            else:
                t0 = time.perf_counter()
                self._ensure_matcher().set_template_cache(self._templates)
                self._log(
                    f"预计算模板特征完成: count={len(self._templates)}, "
                    f"耗时={(time.perf_counter() - t0) * 1000:.1f}ms"
                )

    def _load_keyframe_image(self, keyframe_id: str) -> Optional[np.ndarray]:
        if keyframe_id in self._image_cache:
            return self._image_cache[keyframe_id]
        kf = self.raw.keyframes.get(keyframe_id)
        if kf is None:
            return None
        path = self._session_dir / kf.path
        if not path.exists():
            return None
        try:
            buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            self._image_cache[keyframe_id] = img
            return img
        except Exception:
            return None

    # ------------------------------------------------------------------
    # 正向 bar_slots 状态机
    # ------------------------------------------------------------------
    def _build_initial_bar_state(self):
        """从初始 TEAM_BAR 建立第一张 bar_slots 状态。"""
        t0 = time.perf_counter()
        team_bar_kf = next(
            (kf for kf in self.raw.keyframes.values() if kf.type == KeyframeType.TEAM_BAR),
            None,
        )
        if team_bar_kf is None:
            self._log("未找到 TEAM_BAR 关键帧，无法建立初始 bar 状态")
            return
        bar_img = self._load_keyframe_image(team_bar_kf.id)
        if bar_img is None:
            self._log("TEAM_BAR 关键帧图片加载失败")
            return
        self._log(f"[耗时] 加载 TEAM_BAR 图片: {(time.perf_counter()-t0)*1000:.1f}ms")

        total = self.raw.initial_operator_count + self.raw.initial_item_count
        if total <= 0:
            self._log("初始干员/道具数为 0，跳过初始状态")
            return

        t1 = time.perf_counter()
        quantities = self._parse_quantity_strip(bar_img, total, active_slot=None)
        self._prev_bar_state = self._parse_bar_state(
            bar_img,
            total,
            quantities=quantities,
            time_ms=team_bar_kf.time_ms,
            active_slot=None,  # 初始部署区无 active slot，走动态 ROI
        )
        self._log(f"[耗时] _parse_bar_state: {(time.perf_counter()-t1)*1000:.1f}ms")

        # 检测无限道具：初始道具区中无数量角标的 slot。
        # 注意：数量框检测到但数值识别失败（如模型返回 0）的 slot 仍应视为有限道具，
        # 因此同时参考 _parse_quantity_strip 的成功识别结果和数量框检测的候选 slot。
        t1b = time.perf_counter()
        qty_boxes = self._ocr_quantity_boxes(bar_img)
        window_width, _ = self._recover_window_size_from_bar(bar_img)
        candidate_qty_slots = self._assign_quantity_boxes(
            qty_boxes, total, window_width, active_slot=None
        )
        finite_item_count = 0
        for i in range(self.raw.initial_item_count):
            if i in quantities or i in candidate_qty_slots:
                finite_item_count += 1
                continue
            cost = self._ocr_slot_cost(bar_img, i, total)
            if cost is None:
                cost = 0
            name = f"__infinite_item_{i}__"
            slot_avatar = (
                self._prev_bar_state[i].avatar
                if i < len(self._prev_bar_state)
                else None
            )
            self._infinite_items[i] = _InfiniteItem(
                original_bar_index=i,
                expected_cost=cost,
                name=name,
                present=True,
                avatar=slot_avatar,
            )
            self._summon_costs[name] = cost
            self._summon_deploy_counts[name] = 0
            self._log(f"初始部署区 slot={i}: 无限道具 {name}, expected_cost={cost}")
        self._remaining_item_count = finite_item_count

        # 把无限道具 slot 覆盖回 _prev_bar_state
        new_state: List[_SlotState] = []
        for i, slot in enumerate(self._prev_bar_state):
            if i in self._infinite_items and self._infinite_items[i].present:
                item = self._infinite_items[i]
                new_state.append(
                    _SlotState(
                        name=item.name,
                        is_infinite=True,
                        is_summon=True,
                        original_bar_index=item.original_bar_index,
                        cost=item.expected_cost,
                        avatar=slot.avatar,
                    )
                )
            else:
                new_state.append(slot)
        self._prev_bar_state = new_state
        # 清理 _item_bar_index 中因 _parse_bar_state 临时生成的无限道具 __item_N__ 条目，
        # 避免后续按 bar_index 反查时把无限道具当成有限道具。
        for original_idx in self._infinite_items.keys():
            stale_name = self._item_name_for_bar_index(original_idx)
            self._item_bar_index.pop(stale_name, None)
        # 无限道具按原始 bar_index 登记到 _item_bar_index，使构建 script.items 时能与
        # script.summons 使用同一名称（含名称卡 OCR 重命名后）。
        for original_idx, item in self._infinite_items.items():
            self._item_bar_index[item.name] = original_idx
        self._log(
            f"[耗时] 检测无限道具: {(time.perf_counter()-t1b)*1000:.1f}ms, "
            f"finite={finite_item_count}, infinite={len(self._infinite_items)}"
        )

        t2 = time.perf_counter()
        for i, slot in enumerate(self._prev_bar_state):
            if not slot.is_item and not slot.is_summon:
                self._operator_index[slot.name] = i
                self._operator_info[slot.name] = slot
                # 初始部署区识别干员费用，供后续费用排序使用
                cost = self._ocr_slot_cost(bar_img, i, total)
                if cost is not None:
                    slot.cost = cost
        # 保存初始干员从左到右的视觉顺序，后续构建 script.operators 直接回填
        self._initial_operator_order = [
            s.name
            for s in reversed(self._prev_bar_state)
            if not s.is_item and not s.is_summon
        ]
        self._log(f"[耗时] 初始费用 OCR({len(self._prev_bar_state)} slots): {(time.perf_counter()-t2)*1000:.1f}ms")
        # 初始状态按截图实际位置保留，不按费用排序（同费用干员顺序会乱）
        self._log(
            f"初始 bar 状态: total={total}, finite_items={self._remaining_item_count}, "
            f"infinite_items={len(self._infinite_items)}, "
            f"slots={[f'{s.name}(cost={s.cost},qty={s.quantity},inf={s.is_infinite})' for s in self._prev_bar_state]}"
        )
        self._log(f"[耗时] 构建初始部署区状态: {(time.perf_counter()-t0)*1000:.1f}ms")

    def _apply_initial_bar_state(self):
        """直接使用执行器导出的部署栏状态作为初始 bar 状态。

        这样可以避免录制器把脚本执行后剩余的干员/道具当成 unknown 或召唤物 obs，
        保证后续用户录制的第一条 DEPLOY/RETREAT/SKILL 能解析到正确名称。
        """
        t0 = time.perf_counter()
        state_dicts = self._initial_bar_state or []
        slots: List[_SlotState] = []
        self._remaining_item_count = 0
        for i, d in enumerate(state_dicts):
            orig_idx = d.get("original_bar_index", i)
            slot = _SlotState(
                name=d.get("name", f"__unknown_{i}__"),
                is_item=bool(d.get("is_item", False)),
                is_summon=bool(d.get("is_summon", False)),
                is_infinite=bool(d.get("is_infinite", False)),
                original_bar_index=orig_idx,
                quantity=d.get("quantity"),
                cost=d.get("cost"),
                avatar=None,
            )
            slots.append(slot)
            if slot.is_infinite:
                # 无限道具：按召唤物处理，但不计入普通道具数量
                self._infinite_items[orig_idx] = _InfiniteItem(
                    original_bar_index=orig_idx,
                    expected_cost=slot.cost or 0,
                    name=slot.name,
                    present=True,
                    avatar=None,
                )
                self._summon_costs[slot.name] = slot.cost or 0
                self._summon_deploy_counts[slot.name] = 0
                self._item_bar_index[slot.name] = orig_idx
            elif slot.is_item:
                self._item_bar_index[slot.name] = orig_idx
                self._item_initial_quantity[slot.name] = (
                    slot.quantity if slot.quantity is not None else 1
                )
                self._remaining_item_count += 1
            elif slot.is_summon:
                self._summon_costs[slot.name] = slot.cost or 0
                self._summon_deploy_counts[slot.name] = 0
            else:
                self._operator_index[slot.name] = i
                self._operator_info[slot.name] = slot
        self._prev_bar_state = slots
        self._initial_operator_order = [
            s.name for s in reversed(slots)
            if not s.is_item and not s.is_summon
        ]
        self._log(
            f"应用脚本执行后的部署区状态: total={len(slots)}, "
            f"items={self._remaining_item_count}, "
            f"operators={len(self._operator_index)}, "
            f"summons={len(self._summon_costs)}, "
            f"slots={[f'{s.name}(cost={s.cost},qty={s.quantity})' for s in slots]}"
        )
        self._log(f"[耗时] 应用初始部署区状态: {(time.perf_counter()-t0)*1000:.1f}ms")

    def _record_initial_summon_quantities(self):
        """在正向处理前记录初始部署栏中各召唤物的数量，供后续 ADD_SUMMON 修剪使用。"""
        self._initial_summon_quantities = {
            slot.name: slot.quantity
            for slot in self._prev_bar_state
            if slot.is_summon and not slot.is_infinite and slot.quantity is not None
        }
        if self.debug and self._initial_summon_quantities:
            self._log(
                f"初始召唤物数量: "
                f"{[(n, q) for n, q in self._initial_summon_quantities.items()]}"
            )

    # ------------------------------------------------------------------
    # 无限道具处理
    # ------------------------------------------------------------------
    def _current_bar_index_for_infinite(self, original_bar_index: int) -> int:
        """计算指定原始 bar_index 的无限道具在当前部署栏上的 bar_index。"""
        count = 0
        for slot in self._prev_bar_state:
            if slot.is_item and slot.name.startswith("__item_"):
                idx = int(slot.name.split("_")[3])
                if idx < original_bar_index:
                    count += 1
        for idx, item in self._infinite_items.items():
            if idx < original_bar_index and item.present:
                count += 1
        return count

    def _item_name_at_current_index(self, current_index: int) -> str:
        """根据当前 bar_index 返回道具名称（基于原始 bar_index 顺序）。"""
        # 如果当前索引已有已知道具名称，直接复用，避免把实际道具名覆盖成 __item_N__
        if (
            0 <= current_index < len(self._prev_bar_state)
            and self._prev_bar_state[current_index].is_item
            and not self._prev_bar_state[current_index].name.startswith("__item_")
        ):
            return self._prev_bar_state[current_index].name

        present_original: List[Tuple[int, bool]] = []
        for slot in self._prev_bar_state:
            if slot.is_item and slot.name.startswith("__item_"):
                present_original.append((int(slot.name.split("_")[3]), False))
        for original_idx, item in self._infinite_items.items():
            if item.present:
                present_original.append((original_idx, True))
        present_original.sort(key=lambda x: x[0])
        if 0 <= current_index < len(present_original):
            orig_idx, is_inf = present_original[current_index]
            if is_inf:
                return self._infinite_items[orig_idx].name
            return self._item_name_for_bar_index(orig_idx)
        return self._item_name_for_bar_index(current_index)

    def _build_infinite_slot_map(self) -> Dict[int, _InfiniteItem]:
        """返回当前 bar_index -> 无限道具 的映射。"""
        result: Dict[int, _InfiniteItem] = {}
        for item in self._infinite_items.values():
            if item.present:
                cur_idx = self._current_bar_index_for_infinite(item.original_bar_index)
                result[cur_idx] = item
        return result

    def _ocr_cost_value_with_conf(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int] = None,
    ) -> Optional[Tuple[int, float]]:
        """识别指定 slot 的费用并返回 (value, confidence)，优先 ONNX 模型。"""
        cost_crop = self._crop_cost_from_bar(
            bar_img, bar_index, total_slots, active_slot=active_slot
        )
        if cost_crop is None or cost_crop.size == 0:
            return None
        try:
            proc_inv = cost_recognition.preprocess_cost_image_inv(cost_crop)
            model_result = self._digit_recognizer.predict_cost(proc_inv)
            if model_result:
                value, conf = model_result
                if 0 <= value <= 99:
                    return value, conf
        except Exception:
            pass
        if self.ocr is None:
            return None
        for preprocess in (
            cost_recognition.preprocess_cost_image,
            cost_recognition.preprocess_cost_image_inv,
        ):
            try:
                proc = preprocess(cost_crop)
                result = cost_recognition.extract_cost_with_conf(
                    self.ocr.recognize(proc, min_confidence=0.5), min_conf=0.5
                )
                if result:
                    return result[0], result[1]
            except Exception:
                continue
        return None

    def _update_infinite_items_presence(
        self, bar_img: np.ndarray, time_ms: int, active_slot: Optional[int] = None
    ):
        """根据当前 bar 图检测每个无限道具是否在场，并同步 _prev_bar_state。"""
        if not self._infinite_items:
            return
        total_base = len(self._prev_bar_state)
        changed = False
        for original_idx, item in self._infinite_items.items():
            cur_idx = self._current_bar_index_for_infinite(original_idx)
            candidates = [total_base]
            if total_base > 12:
                candidates.append(total_base + 1)
            best_result: Optional[Tuple[int, float]] = None
            for total in candidates:
                result = self._ocr_cost_value_with_conf(
                    bar_img, cur_idx, total, active_slot=active_slot
                )
                if result is None:
                    continue
                if best_result is None or result[1] > best_result[1]:
                    best_result = result
            if best_result is None:
                continue
            value, conf = best_result

            # 调试：保存本次无限道具费用 ROI，便于排查低置信度误报
            if self.debug:
                best_total = total_base
                if total_base > 12:
                    # 若 best_result 来自 total_base+1，重新裁剪以匹配
                    res_total = total_base
                    for total in candidates:
                        tmp = self._ocr_cost_value_with_conf(
                            bar_img, cur_idx, total, active_slot=active_slot
                        )
                        if tmp is not None and tmp[0] == value and tmp[1] == conf:
                            res_total = total
                            break
                    best_total = res_total
                cost_crop = self._crop_cost_from_bar(
                    bar_img, cur_idx, best_total, active_slot=active_slot
                )
                if cost_crop is not None and cost_crop.size > 0:
                    self._save_ocr_debug(
                        cost_crop,
                        f"infinite_{item.name}_{time_ms}_idx{cur_idx}_total{best_total}",
                        value=value,
                        conf=conf,
                    )

            # 低置信度结果不用于切换在场状态，避免单次误报导致道具被错误移除
            if conf < self._INFINITE_ITEM_PRESENCE_CONF_THRESHOLD:
                self._log(
                    f"无限道具 {item.name} 在场检测置信度过低，忽略本次结果 "
                    f"(OCR={value}, expected={item.expected_cost}, conf={conf:.2f}, cur_idx={cur_idx})"
                )
                continue

            # 费用相同时，进一步用头像相似度或红像素判断，避免费用相同的干员被误判为道具
            best_total = total_base
            if total_base > 12:
                res_total = total_base
                for total in candidates:
                    tmp = self._ocr_cost_value_with_conf(
                        bar_img, cur_idx, total, active_slot=active_slot
                    )
                    if tmp is not None and tmp[0] == value and tmp[1] == conf:
                        res_total = total
                        break
                best_total = res_total

            current_avatar = self._crop_slot_avatar(
                bar_img, cur_idx, best_total, active_slot=active_slot
            )
            avatar_sim = 0.0
            red_ratio = 0.0
            if current_avatar is not None and current_avatar.size > 0:
                red_ratio = self._compute_red_ratio(current_avatar)
                if item.avatar is not None and self._matcher is not None:
                    try:
                        matrix = self._matcher.compute_score_matrix(
                            {item.name: item.avatar}, [current_avatar]
                        )
                        avatar_sim = matrix.get(item.name, {}).get(0, 0.0)
                    except Exception as e:
                        self._log(f"无限道具 {item.name} 头像匹配异常: {e}")

            cost_match = value == item.expected_cost
            # 没有初始头像时退回到纯费用判断
            visual_match = (
                item.avatar is None
                or current_avatar is None
                or avatar_sim >= 0.8
                or red_ratio >= self._RED_COOLDOWN_MIN_RATIO
            )
            new_present = cost_match and visual_match

            if self.debug or new_present != item.present:
                self._log(
                    f"无限道具 {item.name} 在场检测: OCR={value}, expected={item.expected_cost}, "
                    f"conf={conf:.2f}, cur_idx={cur_idx}, avatar_sim={avatar_sim:.2f}, "
                    f"red_ratio={red_ratio:.3f}, present={new_present}"
                )
                if current_avatar is not None and current_avatar.size > 0:
                    self._save_ocr_debug(
                        current_avatar,
                        f"infinite_avatar_{item.name}_{time_ms}_idx{cur_idx}_total{best_total}",
                        value=value,
                        conf=conf,
                    )

            was_present = item.present
            item.present = new_present
            if item.present != was_present:
                changed = True
                if item.present:
                    # 返回部署栏，生成 ADD_SUMMON（与当前 DEPLOY 同帧，按列表顺序先于 DEPLOY）
                    self._actions.append(
                        OperatorAction(
                            time_ms=time_ms,
                            action=ActionType.ADD_SUMMON,
                            operator_name=item.name,
                            grid=(1, 0),
                        )
                    )
                    self._log(
                        f"无限道具 {item.name} 返回部署栏，ADD_SUMMON @ {time_ms}ms "
                        f"(OCR={value}, expected={item.expected_cost}, conf={conf:.2f}, cur_idx={cur_idx})"
                    )
                else:
                    self._log(
                        f"无限道具 {item.name} 离开部署栏 "
                        f"(OCR={value}, expected={item.expected_cost}, conf={conf:.2f}, cur_idx={cur_idx})"
                    )
        if changed:
            self._rebuild_prev_bar_state_items()

    def _rebuild_prev_bar_state_items(self):
        """根据当前有限/无限道具状态重建 _prev_bar_state 的道具区。"""
        non_items = [s for s in self._prev_bar_state if not s.is_item and not s.is_infinite]
        finite_slots: List[_SlotState] = []
        for slot in self._prev_bar_state:
            if slot.is_item and slot.name.startswith("__item_"):
                finite_slots.append(slot)
        infinite_slots: List[_SlotState] = []
        for original_idx in sorted(self._infinite_items.keys()):
            item = self._infinite_items[original_idx]
            if item.present:
                infinite_slots.append(
                    _SlotState(
                        name=item.name,
                        is_infinite=True,
                        is_summon=True,
                        original_bar_index=item.original_bar_index,
                        cost=item.expected_cost,
                    )
                )

        def _orig_idx(s: _SlotState) -> int:
            if s.original_bar_index is not None:
                return s.original_bar_index
            if s.name.startswith("__item_"):
                return int(s.name.split("_")[3])
            return 0

        item_slots = finite_slots + infinite_slots
        item_slots.sort(key=_orig_idx)
        self._remaining_item_count = len(finite_slots)
        self._prev_bar_state = item_slots + non_items
        self._log(
            f"重建 bar 状态: finite={self._remaining_item_count}, "
            f"infinite={len(infinite_slots)}, non_items={len(non_items)}, "
            f"items={[s.name for s in item_slots]}"
        )

    def _preprocess_quantity_strip(self, strip: np.ndarray, invert: bool = False) -> np.ndarray:
        """对全宽数量条做二值化预处理，不放大。

        Args:
            invert: True 时使用高阈值反色：仅纯白像素转为黑色，其余全部转为白色，
                    避免图标边缘的浅灰色被 OTSU 误判为文字；False 时使用固定阈值
                    （白字黑底）。
        """
        gray = cv2.cvtColor(strip, cv2.COLOR_BGRA2GRAY)
        if invert:
            # 将较亮像素（>= threshold）→ 黑，其他 → 白
            _, binary = cv2.threshold(
                gray,
                constants.DEPLOY_BAR_QUANTITY_INVERT_THRESHOLD,
                255,
                cv2.THRESH_BINARY_INV,
            )
        else:
            _, binary = cv2.threshold(
                gray,
                constants.DEPLOY_BAR_QUANTITY_WHITE_THRESHOLD,
                255,
                cv2.THRESH_BINARY,
            )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _extract_quantity_from_ocr(
        results: list, min_conf: float = 0.5
    ) -> Optional[Tuple[int, float, str]]:
        """从 OCR 结果中提取数量数字，并返回原始识别文本供调试。

        数量角标常见格式为 "x1" / "X1" / "×1" / "*1" 等，
        这里先去掉左侧常见乘号/空白，再提取纯数字。
        """
        best_qty = None
        best_conf = 0.0
        best_raw = ""
        for bbox, (text, conf) in results:
            if conf < min_conf:
                continue
            cleaned = re.sub(r"^[xX×*\s]+", "", text)
            digits = "".join(c for c in cleaned if c.isdigit())
            if not digits:
                continue
            try:
                qty = int(digits)
            except ValueError:
                continue
            if qty >= 100:
                continue
            if conf > best_conf:
                best_conf = conf
                best_qty = qty
                best_raw = text
        return (best_qty, best_conf, best_raw) if best_qty is not None else None

    def _ocr_quantity_boxes(self, bar_img: np.ndarray) -> List[Tuple[List, float]]:
        """对全宽数量条做检测，优先使用 YOLO，YOLO 不可用时回退到 Paddle detect_text。

        YOLO 在 binarized 数量条上直接检测 X+数字 角标框，避免 Paddle 检测漏框
        以及与 ONNX 模型并发时的抖动问题。
        最终框的位置仍交给 slot 级别的标定 ROI + X_num_model 读数。
        """
        img_id = id(bar_img)
        if img_id in self._quantity_boxes_cache:
            return self._quantity_boxes_cache[img_id]

        boxes: List[Tuple[List, float]] = []
        if bar_img.size == 0:
            self._quantity_boxes_cache[img_id] = boxes
            return boxes

        h_bar, w_bar = bar_img.shape[:2]
        window_width, window_height = self._recover_window_size_from_bar(bar_img)
        bar_top = window_height * self._BAR_CAPTURE_TOP_RATIO
        y1 = int(round(window_height * self._QUANTITY_ROI_Y_RATIO - bar_top))
        y2 = y1 + int(round(window_height * self._QUANTITY_ROI_H_RATIO))
        y1 = max(0, min(h_bar - 1, y1))
        y2 = max(y1 + 1, min(h_bar, y2))
        strip = bar_img[y1:y2, 0:w_bar]
        if strip.size == 0:
            self._quantity_boxes_cache[img_id] = boxes
            return boxes

        all_boxes: List[Tuple[List, float]] = []
        debug_canvas: Optional[np.ndarray] = None
        use_yolo = False

        # 1) 优先使用 YOLO 检测
        try:
            proc = self._preprocess_quantity_strip(strip, invert=True)
            if self._yolo_detector is not None and self._yolo_detector.available:
                yolo_boxes = self._yolo_detector.detect(proc)
                use_yolo = True
                if self.debug:
                    debug_canvas = proc.copy()
                for bbox, det_conf in yolo_boxes:
                    if not bbox:
                        continue
                    xs = [p[0] for p in bbox]
                    center_x = sum(xs) / len(xs)
                    all_boxes.append((bbox, center_x))
                    self._log(
                        f"YOLO 数量框: box_x=[{min(xs):.0f},{max(xs):.0f}] "
                        f"det_conf={det_conf:.2f}"
                    )
                    if debug_canvas is not None:
                        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(debug_canvas, [pts], True, (0, 255, 0), 1)
        except Exception as e:
            self._log(f"YOLO 数量条检测异常，回退到 Paddle: {e}")
            use_yolo = False

        # 2) YOLO 不可用时回退到 Paddle detect_text + 框内 OCR 过滤
        if not use_yolo:
            if self.ocr is None:
                self._quantity_boxes_cache[img_id] = boxes
                return boxes
            for invert in (True,):
                try:
                    proc = self._preprocess_quantity_strip(strip, invert=invert)
                    bboxes = self.ocr.detect_text(proc) if self.ocr is not None else []
                except Exception as e:
                    self._log(f"数量条检测异常 (invert={invert}): {e}")
                    continue
                if self.debug:
                    debug_canvas = proc.copy()
                for bbox, det_conf in bboxes:
                    if not bbox:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x1, x2 = max(0, int(min(xs))), min(w_bar, int(max(xs)))
                    y1b, y2b = max(0, int(min(ys))), min(strip.shape[0], int(max(ys)))
                    if x2 <= x1 or y2b <= y1b:
                        continue
                    # 按 1600h 归一化尺寸过滤过小/过大的检测框
                    scale = window_height / 1600
                    w_norm = (x2 - x1) / scale
                    h_norm = (y2b - y1b) / scale
                    if (
                        w_norm < self._QUANTITY_MIN_W_PX_1600
                        or h_norm < self._QUANTITY_MIN_H_PX_1600
                    ):
                        self._log(
                            f"数量条检测忽略过小框: "
                            f"box_x=[{x1:.0f},{x2:.0f}] det_conf={det_conf:.2f} "
                            f"(w={w_norm:.1f}, h={h_norm:.1f})"
                        )
                        continue

                    # 对候选框做 OCR 识别，只保留包含数字的数量框
                    center_x = sum(xs) / len(xs)
                    try:
                        crop = proc[y1b:y2b, x1:x2]
                        ocr_results = self.ocr.recognize(crop, min_confidence=0.5)
                        extracted = self._extract_quantity_from_ocr(
                            ocr_results, min_conf=0.5
                        )
                    except Exception as e:
                        self._log(
                            f"数量框 OCR 识别异常 box_x=[{x1:.0f},{x2:.0f}]: {e}"
                        )
                        extracted = None

                    if extracted is None:
                        # 记录原始 OCR 文本，便于分析误过滤
                        raw_texts = [f"'{text}'({conf:.2f})" for _, (text, conf) in ocr_results]
                        self._log(
                            f"数量框过滤: box_x=[{x1:.0f},{x2:.0f}] "
                            f"det_conf={det_conf:.2f} 未识别到数字 "
                            f"raw_ocr={raw_texts}"
                        )
                        continue

                    qty, rec_conf, raw_text = extracted
                    self._log(
                        f"数量框保留: box_x=[{x1:.0f},{x2:.0f}] "
                        f"det_conf={det_conf:.2f} OCR='{raw_text}' "
                        f"qty={qty} rec_conf={rec_conf:.2f}"
                    )
                    all_boxes.append((bbox, center_x))
                    if debug_canvas is not None:
                        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(debug_canvas, [pts], True, (0, 255, 0), 1)

        # 按 center_x 距离合并相近的框
        bucket_w = max(10, window_width / 36)
        merge_threshold = bucket_w / 2

        merged: List[Tuple[List, float]] = []
        for bbox, center_x in all_boxes:
            found = False
            for i, (_mbbox, mcenter_x) in enumerate(merged):
                if abs(center_x - mcenter_x) < merge_threshold:
                    found = True
                    break
            if not found:
                merged.append((bbox, center_x))

        boxes = sorted(merged, key=lambda x: x[1])

        if self.debug:
            self._quantity_strip_debug_counter += 1
            tag = "yolo" if use_yolo else "paddle"
            if debug_canvas is not None:
                try:
                    self._save_ocr_debug(
                        debug_canvas,
                        f"quantity_strip_{self._quantity_strip_debug_counter:04d}_{tag}",
                        value=len(all_boxes),
                    )
                except Exception as e:
                    self._log(f"保存数量条调试图失败 ({tag}): {e}")

            # 保存合并后的最终结果
            if debug_canvas is not None:
                try:
                    merged_canvas = debug_canvas.copy()
                    for bbox, _center_x in boxes:
                        pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(merged_canvas, [pts], True, (0, 0, 255), 1)
                    self._save_ocr_debug(
                        merged_canvas,
                        f"quantity_strip_{self._quantity_strip_debug_counter:04d}_merged",
                        value=len(boxes),
                    )
                except Exception as e:
                    self._log(f"保存数量条合并调试图失败: {e}")

        self._quantity_boxes_cache[img_id] = boxes
        return boxes

    def _assign_quantity_boxes(
        self,
        boxes: List[Tuple[List, float]],
        total_slots: int,
        window_width: int,
        active_slot: Optional[int] = None,
    ) -> Set[int]:
        """根据 bbox 与每个 slot 数量 ROI 的重叠，把数量框映射为候选 slot_index。

        优先使用 data/quantity_roi_config_total{total_slots}.json 中的标定 x 范围；
        无标定时回退到均匀槽宽估算。
        只返回可能存在数量的 slot 集合，具体数值由调用方用标定 ROI + X_num_model 读取。
        """
        result: Set[int] = set()
        calibrated = self._get_quantity_roi_xranges(total_slots, active_slot=active_slot)
        use_calibration = calibrated is not None and len(calibrated) >= total_slots
        if use_calibration:
            self._log("  数量框分配使用标定 ROI")
        cell_w = window_width / 12 if total_slots <= 12 else window_width / total_slots
        for bbox, _center_x in boxes:
            xs = [p[0] for p in bbox]
            box_x1, box_x2 = min(xs), max(xs)
            best_idx = -1
            best_overlap = 0.0
            for i in range(total_slots):
                if use_calibration and i in calibrated:
                    x1_ratio, x2_ratio = calibrated[i]
                    x1_i = window_width * x1_ratio
                    x2_i = window_width * x2_ratio
                else:
                    # slot i 的数量 ROI：slot 中心线到右边界
                    x1_i = window_width - cell_w * (i + 0.5)
                    x2_i = window_width - cell_w * i
                overlap = max(0.0, min(box_x2, x2_i) - max(box_x1, x1_i))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            if best_idx < 0:
                self._log(
                    f"  数量框分配忽略: box_x=[{box_x1:.0f},{box_x2:.0f}] 无重叠 slot"
                )
                continue
            self._log(
                f"  数量框分配: box_x=[{box_x1:.0f},{box_x2:.0f}] -> slot[{best_idx}]"
            )
            result.add(best_idx)
        return result

    def _parse_quantity_strip(
        self,
        bar_img: np.ndarray,
        total_slots: int,
        boxes: Optional[List[Tuple[List, float]]] = None,
        active_slot: Optional[int] = None,
    ) -> Dict[int, int]:
        """对数量条 OCR，返回 slot_index -> quantity。

        先用全宽数量条检测确定哪些 slot 可能存在数量，再对这些 slot 用标定 ROI
        单独裁剪，走 X_num_model 读取具体数值。
        """
        t0 = time.perf_counter()
        if bar_img.size == 0:
            return {}
        window_width, _ = self._recover_window_size_from_bar(bar_img)
        if boxes is None:
            boxes = self._ocr_quantity_boxes(bar_img)
        self._log(f"[耗时] _ocr_quantity_boxes: {(time.perf_counter()-t0)*1000:.1f}ms")
        t1 = time.perf_counter()
        candidate_slots = self._assign_quantity_boxes(boxes, total_slots, window_width, active_slot=active_slot)

        final_quantities: Dict[int, int] = {}
        for slot_idx in candidate_slots:
            refined = self._recognize_quantity_from_bar_img(
                bar_img, slot_idx, total_slots, active_slot=active_slot
            )
            if refined is not None and 0 < refined < 100:
                final_quantities[slot_idx] = refined
                self._log(f"  quantity slot={slot_idx}: 标定ROI模型={refined}")
        self._log(f"[耗时] _recognize_quantity per-slot: {(time.perf_counter()-t1)*1000:.1f}ms")
        return final_quantities

    def _infer_total_from_quantity(
        self,
        bar_img: np.ndarray,
        prev_total: int,
        prev_qty_count: int,
        action_delta_ops: int = 0,
    ) -> Tuple[int, List[Tuple[List, float]]]:
        """根据数量条变化推断当前总槽位数，并返回识别到的数量框供后续复用。

        思路：数量框个数 = 道具槽 + 召唤物槽。前后数量框个数的变化即道具/召唤物槽的变化；
        干员部署/撤退会单独改变干员槽数量。因此：
          total = prev_total + action_delta_ops + (cur_qty_count - prev_qty_count)
        其中 action_delta_ops: 部署干员=-1，撤退干员=+1，道具/召唤物部署=0。
        """
        if bar_img.size == 0:
            return max(1, prev_total + action_delta_ops), []
        t_detect = time.perf_counter()
        boxes = self._ocr_quantity_boxes(bar_img)
        self._log(
            f"[耗时] _ocr_quantity_boxes (infer_total): "
            f"{(time.perf_counter() - t_detect) * 1000:.1f}ms, boxes={len(boxes)}"
        )
        cur_qty_count = len(boxes)
        delta_qty = cur_qty_count - prev_qty_count
        total = max(1, prev_total + action_delta_ops + delta_qty)
        self._log(
            f"数量条推断 total: prev_total={prev_total}, prev_qty={prev_qty_count}, "
            f"cur_qty={cur_qty_count}, delta_ops={action_delta_ops}, result={total}"
        )
        return total, boxes

    def _crop_slot_avatar(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        y_shift: int = 0,
        active_slot: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """从整栏图中裁剪出指定 slot 的头像；若带位移裁剪失败则回退到无位移。

        当 total_slots > 12 且 active_slot 等于当前 bar_index 时，按标定工具规则
        整体向上移动 _ACTIVE_SELF_Y_SHIFT_PX 像素。
        """
        effective_y_shift = y_shift
        if (
            active_slot is not None
            and active_slot == bar_index
            and total_slots > 12
        ):
            effective_y_shift -= self._ACTIVE_SELF_Y_SHIFT_PX
        crop = self._crop_slot_from_bar(
            bar_img, bar_index, total_slots, y_shift=effective_y_shift, active_slot=active_slot
        )
        if crop is None and effective_y_shift != 0:
            crop = self._crop_slot_from_bar(
                bar_img, bar_index, total_slots, y_shift=0, active_slot=active_slot
            )
        return crop

    def _compute_avatar_y_shift(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int],
    ) -> int:
        """计算头像裁剪所需的垂直位移。

        total_slots <= 12 时，active slot 整体偏高 _ACTIVE_SELF_Y_SHIFT_PX 像素；
        total_slots > 12 时，active slot 的形变由 _crop_slot_avatar 统一按标定工具规则
        处理，本函数不再重复计算。
        """
        if active_slot is None or bar_index != active_slot:
            return 0
        if total_slots <= 12:
            return -self._ACTIVE_SELF_Y_SHIFT_PX
        return 0

    def _ocr_slot_cost(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        y_shift: int = 0,
        active_slot: Optional[int] = None,
    ) -> Optional[int]:
        """对指定 slot 裁剪费用 ROI 并识别。优先使用 ONNX 数字模型。"""
        t_total = time.perf_counter()
        cache_key = (id(bar_img), bar_index, total_slots, y_shift, active_slot)
        if cache_key in self._ocr_cost_cache:
            return self._ocr_cost_cache[cache_key]

        t_crop = time.perf_counter()
        cost_crop = self._crop_cost_from_bar(
            bar_img, bar_index, total_slots, y_shift=y_shift, active_slot=active_slot
        )
        self._log(f"[耗时] cost crop slot={bar_index}: {(time.perf_counter()-t_crop)*1000:.2f}ms")
        if cost_crop is None or cost_crop.size == 0:
            self._ocr_cost_cache[cache_key] = None
            return None

        result = None

        # 1) 优先走 ONNX 数字模型（黑字白底）
        if self._digit_recognizer is not None:
            try:
                t_pre = time.perf_counter()
                proc_inv = cost_recognition.preprocess_cost_image_inv(cost_crop)
                self._log(f"[耗时] cost preprocess slot={bar_index}: {(time.perf_counter()-t_pre)*1000:.2f}ms")
                t0 = time.perf_counter()
                model_result = self._digit_recognizer.predict_cost(proc_inv)
                model_ms = (time.perf_counter() - t0) * 1000
                if model_result:
                    value, conf = model_result
                    if 0 <= value <= 99:
                        self._log(
                            f"  OCR cost slot={bar_index}: 模型={value} "
                            f"(conf={conf:.2f}, 耗时={model_ms:.2f}ms)"
                        )
                        result = (value, conf)
            except Exception as e:
                self._log(f"  OCR cost slot={bar_index} 模型异常: {e}")

        # 2) 模型失败则 fallback 到 OCR 双路
        if result is None and self.ocr is not None:
            for preprocess_name, preprocess in (
                ("固定阈值", cost_recognition.preprocess_cost_image),
                ("反色", cost_recognition.preprocess_cost_image_inv),
            ):
                try:
                    proc = preprocess(cost_crop)
                    result = cost_recognition.extract_cost_with_conf(
                        self.ocr.recognize(proc, min_confidence=0.5), min_conf=0.5
                    )
                    if result:
                        self._log(
                            f"  OCR cost slot={bar_index}: {result[0]} "
                            f"(conf={result[1]:.2f}, 方式={preprocess_name})"
                        )
                        break
                except Exception as e:
                    self._log(f"  OCR cost slot={bar_index} 异常 ({preprocess_name}): {e}")
                    continue

        if self.debug:
            t_save = time.perf_counter()
            shift_tag = f"_shift{y_shift}" if y_shift else ""
            self._save_ocr_debug(
                cost_crop,
                f"cost_slot_{bar_index}_total_{total_slots}{shift_tag}",
                value=result[0] if result else None,
                conf=result[1] if result else None,
            )
            self._log(f"[耗时] cost save debug slot={bar_index}: {(time.perf_counter()-t_save)*1000:.2f}ms")

        value = result[0] if result else None
        self._ocr_cost_cache[cache_key] = value
        self._log(f"[耗时] cost slot={bar_index} total: {(time.perf_counter()-t_total)*1000:.2f}ms")
        return value

    def _parse_bar_state(
        self,
        bar_img: np.ndarray,
        total_slots: int,
        quantities: Optional[Dict[int, int]] = None,
        time_ms: int = 0,
        active_slot: Optional[int] = None,
    ) -> List[_SlotState]:
        """把 post-deploy 整栏图解析为 bar_slots 列表。

        召唤物不再做跨帧模板匹配，而是分配唯一 obs_id，最后统一聚类。
        """
        t_parse_qty = time.perf_counter()
        if quantities is None:
            quantities = self._parse_quantity_strip(
                bar_img, total_slots, active_slot=active_slot
            )
        self._log(f"[耗时] _parse_quantity_strip: {(time.perf_counter()-t_parse_qty)*1000:.1f}ms")
        t_avatar = time.perf_counter()
        infinite_slot_map = self._build_infinite_slot_map()
        item_region_size = self._remaining_item_count + len(infinite_slot_map)
        states: List[_SlotState] = []
        for i in range(total_slots):
            qty = quantities.get(i)
            y_shift = self._compute_avatar_y_shift(bar_img, i, total_slots, active_slot)
            if i in infinite_slot_map:
                item = infinite_slot_map[i]
                avatar = self._crop_slot_avatar(
                    bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                )
                states.append(
                    _SlotState(
                        name=item.name,
                        is_infinite=True,
                        is_summon=True,
                        original_bar_index=item.original_bar_index,
                        cost=item.expected_cost,
                        avatar=avatar,
                    )
                )
            elif qty is not None:
                if i < item_region_size:
                    name = self._item_name_at_current_index(i)
                    avatar = self._crop_slot_avatar(
                        bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                    )
                    states.append(
                        _SlotState(
                            name=name,
                            is_item=True,
                            quantity=qty,
                            avatar=avatar,
                        )
                    )
                    self._item_initial_quantity[name] = max(
                        self._item_initial_quantity.get(name, 0), qty
                    )
                    self._item_bar_index[name] = i
                else:
                    # 召唤物：分配唯一 obs_id，最后聚类时再决定真实身份
                    shifted_avatar = self._crop_slot_avatar(
                        bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                    )
                    obs_id = self._next_summon_obs_name()
                    cost = self._ocr_slot_cost(
                        bar_img, i, total_slots, active_slot=active_slot
                    )
                    if cost is not None:
                        self._summon_costs[obs_id] = cost
                    self._register_summon_obs(
                        obs_id, shifted_avatar, cost, qty, time_ms, i
                    )
                    states.append(
                        _SlotState(
                            name=obs_id,
                            is_summon=True,
                            quantity=qty,
                            cost=cost,
                            avatar=shifted_avatar,
                        )
                    )
            elif i < item_region_size:
                # 道具区 slot 但数量 OCR 失败，仍按道具处理
                name = self._item_name_at_current_index(i)
                avatar = self._crop_slot_avatar(
                    bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                )
                states.append(
                    _SlotState(name=name, is_item=True, avatar=avatar)
                )
                self._item_bar_index[name] = i
            else:
                avatar = self._crop_slot_avatar(
                    bar_img, i, total_slots, y_shift=y_shift
                )
                name, score = self._match_image(avatar)
                # 避免把曾经识别为召唤物的模板错配给干员
                if name is not None and name in self._summon_costs:
                    name = None
                if name is None or (
                    not name.startswith("__") and score < self._match_threshold
                ):
                    name = self._next_unknown_name()
                    if avatar is not None:
                        self._templates[name] = avatar
                states.append(_SlotState(name=name, avatar=avatar))
        self._log(f"[耗时] _parse_bar_state avatar loop({total_slots} slots): {(time.perf_counter()-t_avatar)*1000:.1f}ms")
        return states

    def _parse_bar_state_aligned(
        self,
        bar_img: np.ndarray,
        total_slots: int,
        prev_state: List[_SlotState],
        deployed_bar_index: Optional[int],
        quantities: Optional[Dict[int, int]] = None,
        time_ms: int = 0,
        active_slot: Optional[int] = None,
    ) -> List[_SlotState]:
        """基于上一状态做索引对齐的 bar 解析，避免每步都匹配所有干员头像。

        规则：
          - 无数量的 slot 直接继承上一状态的干员顺序（去掉本次部署的干员）。
          - 有数量的 slot 按 index 判断是道具还是召唤物；
            召唤物不再做跨帧模板匹配，而是分配唯一 obs_id，最后统一聚类。
        """
        if quantities is None:
            quantities = self._parse_quantity_strip(
                bar_img, total_slots, active_slot=active_slot
            )
        prev_ops = [s for s in prev_state if not s.is_item and not s.is_summon]
        deployed_name = None
        if (
            deployed_bar_index is not None
            and 0 <= deployed_bar_index < len(prev_state)
        ):
            deployed_name = prev_state[deployed_bar_index].name
        op_iter = iter([s for s in prev_ops if s.name != deployed_name])

        infinite_slot_map = self._build_infinite_slot_map()
        item_region_size = self._remaining_item_count + len(infinite_slot_map)
        states: List[_SlotState] = []
        for i in range(total_slots):
            qty = quantities.get(i)
            y_shift = self._compute_avatar_y_shift(bar_img, i, total_slots, active_slot)
            if i in infinite_slot_map:
                item = infinite_slot_map[i]
                avatar = self._crop_slot_avatar(
                    bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                )
                states.append(
                    _SlotState(
                        name=item.name,
                        is_infinite=True,
                        is_summon=True,
                        original_bar_index=item.original_bar_index,
                        cost=item.expected_cost,
                        avatar=avatar,
                    )
                )
            elif qty is not None:
                if i < item_region_size:
                    name = self._item_name_at_current_index(i)
                    avatar = self._crop_slot_avatar(
                        bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                    )
                    states.append(
                        _SlotState(name=name, is_item=True, quantity=qty, avatar=avatar)
                    )
                    self._item_initial_quantity[name] = max(
                        self._item_initial_quantity.get(name, 0), qty
                    )
                    self._item_bar_index[name] = i
                else:
                    # 召唤物：统一分配新 obs_id，最后聚类决定真实身份。
                    # 不继承上一状态，避免脚本接管后把初始部署栏已有的召唤物错当成新召唤物。
                    shifted_avatar = self._crop_slot_avatar(
                        bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                    )
                    obs_id = self._next_summon_obs_name()
                    cost = self._ocr_slot_cost(
                        bar_img, i, total_slots, active_slot=active_slot
                    )
                    if cost is not None:
                        self._summon_costs[obs_id] = cost
                    self._register_summon_obs(
                        obs_id, shifted_avatar, cost, qty, time_ms, i
                    )
                    states.append(
                        _SlotState(
                            name=obs_id,
                            is_summon=True,
                            quantity=qty,
                            cost=cost,
                            avatar=shifted_avatar,
                        )
                    )
            elif i < item_region_size:
                # 道具区 slot 但数量 OCR 失败，仍按道具处理
                name = self._item_name_at_current_index(i)
                avatar = self._crop_slot_avatar(
                    bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                )
                states.append(_SlotState(name=name, is_item=True, avatar=avatar))
                self._item_bar_index[name] = i
            else:
                op = next(op_iter, None)
                if op is not None:
                    states.append(_SlotState(name=op.name, avatar=op.avatar))
                else:
                    # 兜底：状态对不上时做一次头像匹配
                    avatar = self._crop_slot_avatar(
                        bar_img, i, total_slots, y_shift=y_shift, active_slot=active_slot
                    )
                    name, score = self._match_image(avatar)
                    if name is not None and name in self._summon_costs:
                        name = None
                    if name is None or (
                        not name.startswith("__") and score < self._match_threshold
                    ):
                        name = self._next_unknown_name()
                        if avatar is not None:
                            self._templates[name] = avatar
                    states.append(_SlotState(name=name, avatar=avatar))
        return states

    def _find_deployed_slot(
        self,
        prev: List[_SlotState],
        cur: List[_SlotState],
        bar_index: Optional[int],
    ) -> _SlotState:
        """根据前后 bar 状态确定本次部署的是哪个 slot。"""
        cur_names = {s.name for s in cur}
        if bar_index is not None and 0 <= bar_index < len(prev):
            slot = prev[bar_index]
            if slot.name not in cur_names:
                return slot
            # 数量发生变化也视为该 slot 被消耗（道具/召唤物）
            cur_slot = next((s for s in cur if s.name == slot.name), None)
            if (
                cur_slot is not None
                and slot.quantity is not None
                and cur_slot.quantity != slot.quantity
            ):
                return slot
        for slot in prev:
            if slot.name not in cur_names:
                return slot
        if bar_index is not None and 0 <= bar_index < len(prev):
            return prev[bar_index]
        return prev[0]

    def _process_actions_forward(self):
        """按时间顺序处理所有 RawAction，维护 bar_slots 正向状态。"""
        t0 = time.perf_counter()
        for raw in sorted(self.raw.actions, key=lambda a: a.time_ms):
            if raw.action == ActionType.DEPLOY:
                self._process_deploy_forward(raw)
            elif raw.action == ActionType.RETREAT:
                self._process_retreat_forward(raw)
            elif raw.action == ActionType.SKILL:
                self._process_skill_forward(raw)
            else:
                self._log(f"忽略未支持的原始动作: {raw.action}")
        self._log(f"[耗时] 正向处理所有动作: {(time.perf_counter()-t0)*1000:.1f}ms")

    def _process_deploy_forward(self, raw: RawAction):
        """基于 pre-deploy 整栏图处理一次 DEPLOY。

        流程：
          1. 从 pre-deploy 图推断当前总槽位数；
          2. 根据录制时保存的点击比例和实际总槽位数换算出真实 slot 索引；
          3. 解析 pre-deploy 状态，检测新增召唤物/道具；
          4. 直接取出该 slot 作为本次部署对象；
          5. 模拟部署后状态。
        """
        t0 = time.perf_counter()
        bar_img = self._load_bar_image(raw)
        click_ratio = self._parse_click_ratio(raw.target_ref)
        prev = self._prev_bar_state

        if click_ratio is None:
            self._log(f"DEPLOY @ {raw.time_ms} 无法解析点击位置，跳过")
            return

        # 无 pre-deploy 图时，按纯逻辑推导；数量检测现在优先走 YOLO，不再强制要求 OCR
        if bar_img is None:
            initial_total = self.raw.initial_operator_count + self.raw.initial_item_count
            fallback_bar_index = 0
            if initial_total > 0:
                fallback_bar_index = max(
                    0, min(initial_total - 1, int(round(click_ratio * initial_total - 0.5)))
                )
            self._fallback_deploy(raw, fallback_bar_index)
            self._log(f"DEPLOY @ {raw.time_ms} 无 pre-deploy 图，兜底耗时={(time.perf_counter()-t0)*1000:.1f}ms")
            return

        prev_total = len(prev)
        prev_qty_count = self._remaining_item_count + sum(
            1 for s in prev if s.is_summon and s.quantity is not None
        )

        # 0. 更新无限道具在场状态（在推断 total 前，确保 prev_total 正确）
        # 对任意总槽位数都需要先算出 active_slot，因为 total<=12 时被拖拽 slot
        # 的费用/数量 ROI 仍需向上偏移 _ACTIVE_SELF_Y_SHIFT_PX。
        t_inf = time.perf_counter()
        window_width, _ = self._recover_window_size_from_bar(bar_img)
        click_offset_from_right = click_ratio * window_width
        preliminary_active = self._nearest_slot_index(
            click_offset_from_right, prev_total, bar_img
        )
        self._update_infinite_items_presence(
            bar_img, raw.time_ms, active_slot=preliminary_active
        )
        # 状态可能已重建，重新取 prev
        prev = self._prev_bar_state
        prev_total = len(prev)
        prev_qty_count = self._remaining_item_count + sum(
            1 for s in prev if s.is_summon and s.quantity is not None
        )
        self._log(f"  [耗时] 更新无限道具状态: {(time.perf_counter()-t_inf)*1000:.1f}ms")

        # 1. 推断 pre-deploy 实际总槽位（action_delta_ops=0，因为尚未部署）
        t_infer = time.perf_counter()
        actual_total, qty_boxes = self._infer_total_from_quantity(
            bar_img, prev_total, prev_qty_count, action_delta_ops=0
        )
        self._log(f"  [耗时] 推断 total: {(time.perf_counter()-t_infer)*1000:.1f}ms")

        # 2. 用点击比例（相对右边缘）在当前实际总槽位数下换算真实 slot 索引
        window_width, _ = self._recover_window_size_from_bar(bar_img)
        click_offset_from_right = click_ratio * window_width
        actual_bar_index = self._nearest_slot_index(
            click_offset_from_right, actual_total, bar_img
        )
        cell_w = window_width / actual_total
        self._log(
            f"DEPLOY @ {raw.time_ms}: click_ratio={click_ratio:.4f}, "
            f"actual_total={actual_total}, window_width={window_width}, "
            f"click_offset={click_offset_from_right:.2f}, cell_w={cell_w:.2f}, "
            f"nearest_raw={click_offset_from_right/cell_w - 0.5:.2f}, "
            f"actual_bar_index={actual_bar_index}"
        )

        # 3. 解析 pre-deploy 状态（active_slot 即本次被点击拖拽的 slot）
        t_parse = time.perf_counter()
        quantities = self._parse_quantity_strip(
            bar_img,
            actual_total,
            boxes=qty_boxes,
            active_slot=actual_bar_index,
        )
        pre_states = self._parse_bar_state_aligned(
            bar_img,
            actual_total,
            prev,
            deployed_bar_index=None,
            quantities=quantities,
            time_ms=raw.time_ms,
            active_slot=actual_bar_index,
        )
        self._log(f"  [耗时] 解析 pre-deploy 状态: {(time.perf_counter()-t_parse)*1000:.1f}ms")

        if not (0 <= actual_bar_index < len(pre_states)):
            self._log(f"DEPLOY @ {raw.time_ms} actual_bar_index 越界，回退")
            initial_total = self.raw.initial_operator_count + self.raw.initial_item_count
            fallback_bar_index = 0
            if initial_total > 0:
                fallback_bar_index = max(
                    0, min(initial_total - 1, int(round(click_ratio * initial_total - 0.5)))
                )
            self._fallback_deploy(raw, fallback_bar_index)
            self._log(f"DEPLOY @ {raw.time_ms} 越界回退，总耗时={(time.perf_counter()-t0)*1000:.1f}ms")
            return

        deployed = pre_states[actual_bar_index]

        # 保存快照，供后续检测因 RETREAT 导致消失的召唤物
        self._deploy_snapshots.append((raw.time_ms, list(prev), list(pre_states), deployed.name))

        # 保存本次 DEPLOY 关联的名称卡关键帧，后续用于 OCR 重命名道具/召唤物
        name_card_kf = next(
            (
                kid
                for kid in raw.keyframe_ids
                if self.raw.keyframes.get(kid)
                and self.raw.keyframes[kid].type == KeyframeType.DEPLOY_NAME_CARD
            ),
            None,
        )
        if name_card_kf:
            self._name_card_kf_by_name.setdefault(deployed.name, []).append(name_card_kf)

        # 4. 检测新增召唤物/道具，生成 ADD_SUMMON / 记录新道具
        prev_names = {s.name for s in prev}
        for slot in pre_states:
            if slot.name in prev_names:
                continue
            if slot.is_summon:
                charges = slot.quantity if slot.quantity is not None and slot.quantity > 0 else 1
                self._actions.append(
                    OperatorAction(
                        time_ms=raw.time_ms,
                        action=ActionType.ADD_SUMMON,
                        operator_name=slot.name,
                        grid=(charges, 0),
                    )
                )
                self._summon_costs[slot.name] = slot.cost or 0
                self._summon_deploy_counts[slot.name] = self._summon_deploy_counts.get(slot.name, 0)
                self._log(f"ADD_SUMMON {slot.name} x{charges} (cost={slot.cost})")
            elif slot.is_item:
                self._ensure_item(slot.name)
                self._item_initial_quantity[slot.name] = max(
                    self._item_initial_quantity.get(slot.name, 0),
                    slot.quantity or 1,
                )
                self._item_bar_index[slot.name] = self._item_bar_index_for_name(pre_states, slot.name)
                self._log(f"新道具 slot {slot.name} quantity={slot.quantity}")

        # 5. 生成本次 DEPLOY
        t_gen = time.perf_counter()
        if deployed.is_item:
            self._deploy_item(raw, deployed)
        elif deployed.is_summon:
            self._deploy_summon(raw, deployed)
        else:
            self._deploy_operator(raw, deployed)
        self._log(f"  [耗时] 生成 DEPLOY 动作: {(time.perf_counter()-t_gen)*1000:.1f}ms")

        # 7. 模拟部署后状态
        t_sim = time.perf_counter()
        self._prev_bar_state = self._simulate_deploy(pre_states, actual_bar_index)
        self._sort_bar_state()
        self._log(f"  [耗时] 模拟部署后状态: {(time.perf_counter()-t_sim)*1000:.1f}ms")

        if self.debug:
            self._save_deploy_debug(bar_img, actual_total, actual_bar_index, deployed, pre_states)

        self._log(f"DEPLOY @ {raw.time_ms} 总耗时={(time.perf_counter()-t0)*1000:.1f}ms")

    def _fallback_deploy(self, raw: RawAction, bar_index: int):
        """缺少 pre-deploy 图或 OCR 时的兜底：按 hint/占位生成 DEPLOY 并简单更新状态。"""
        prev = self._prev_bar_state
        if 0 <= bar_index < len(prev):
            deployed = prev[bar_index]
            name = deployed.name
            is_item = deployed.is_item
            is_summon = deployed.is_summon
        else:
            is_item = bar_index < self.raw.initial_item_count
            name = (
                self._item_name_for_bar_index(bar_index)
                if is_item
                else self._next_unknown_name()
            )
            deployed = None
            is_summon = False

        if is_item:
            self._deploy_item(raw, deployed or _SlotState(name=name, is_item=True, quantity=1))
        elif is_summon:
            self._deploy_summon(raw, deployed or _SlotState(name=name, is_summon=True, quantity=1))
        else:
            self._deploy_operator(raw, deployed or _SlotState(name=name))

        # 简单移除该 slot
        if 0 <= bar_index < len(prev):
            prev.pop(bar_index)
        self._prev_bar_state = prev
        self._sort_bar_state()

    def _deploy_operator(self, raw: RawAction, deployed: _SlotState):
        self._ensure_operator(deployed.name)
        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.DEPLOY,
                operator_name=deployed.name,
                grid=raw.grid,
                direction=raw.direction,
            )
        )
        if raw.grid:
            self._deployed[raw.grid] = deployed.name
        self._log(f"DEPLOY 干员 {deployed.name} @ {raw.grid}")

    def _deploy_summon(self, raw: RawAction, deployed: _SlotState):
        self._summon_deploy_counts[deployed.name] = self._summon_deploy_counts.get(deployed.name, 0) + 1
        self._summon_costs[deployed.name] = deployed.cost or self._summon_costs.get(deployed.name, 0)
        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.DEPLOY,
                operator_name=deployed.name,
                grid=raw.grid,
                direction=raw.direction,
            )
        )
        if raw.grid:
            self._deployed[raw.grid] = deployed.name
        self._log(f"DEPLOY 召唤物 {deployed.name} @ {raw.grid}")

    def _deploy_item(self, raw: RawAction, deployed: _SlotState):
        self._ensure_item(deployed.name)
        self._item_usage_counts[deployed.name] = self._item_usage_counts.get(deployed.name, 0) + 1
        self._item_initial_quantity[deployed.name] = max(
            self._item_initial_quantity.get(deployed.name, 0),
            deployed.quantity or 1,
        )
        bar_index = self._item_bar_index.get(deployed.name, 0)
        self._item_bar_index[deployed.name] = bar_index
        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.DEPLOY,
                operator_name=deployed.name,
                grid=raw.grid,
                direction=raw.direction,
                is_object=False,
            )
        )
        if raw.grid:
            self._deployed[raw.grid] = deployed.name
        self._log(f"DEPLOY 道具 {deployed.name} @ {raw.grid}")

    def _simulate_deploy(self, pre_states: List[_SlotState], bar_index: int) -> List[_SlotState]:
        """根据 pre-deploy 状态和本次部署的 slot 索引，模拟部署后的 bar 状态。"""
        post_states: List[_SlotState] = []
        for i, slot in enumerate(pre_states):
            if i == bar_index:
                if slot.is_item:
                    new_qty = (slot.quantity or 1) - 1
                    if new_qty > 0:
                        post_states.append(
                            _SlotState(
                                name=slot.name,
                                is_item=True,
                                quantity=new_qty,
                                cost=slot.cost,
                                avatar=slot.avatar,
                            )
                        )
                    else:
                        self._remaining_item_count = max(0, self._remaining_item_count - 1)
                        self._log(f"道具 {slot.name} 耗尽，剩余道具 slot -> {self._remaining_item_count}")
                elif slot.is_infinite:
                    # 无限道具部署后离场，等待下次返回
                    if slot.original_bar_index is not None:
                        item = self._infinite_items.get(slot.original_bar_index)
                        if item is not None:
                            item.present = False
                            self._log(f"无限道具 {item.name} 部署后离场")
                elif slot.is_summon:
                    new_qty = (slot.quantity or 1) - 1
                    if new_qty > 0:
                        post_states.append(
                            _SlotState(
                                name=slot.name,
                                is_summon=True,
                                quantity=new_qty,
                                cost=slot.cost,
                                avatar=slot.avatar,
                            )
                        )
                else:
                    # 干员被移除
                    pass
            else:
                post_states.append(slot)
        return post_states

    def _item_bar_index_for_name(self, states: List[_SlotState], name: str) -> int:
        """从状态列表中查找道具 slot 的 bar_index。"""
        for i, s in enumerate(states):
            if s.is_item and s.name == name:
                return i
        return 0

    def _save_deploy_debug(
        self,
        bar_img: np.ndarray,
        total_slots: int,
        highlight_slot: int,
        deployed: _SlotState,
        pre_states: List[_SlotState],
    ):
        slot_labels = {}
        for idx, slot in enumerate(pre_states):
            parts = []
            display_name = slot.name or "?"
            if len(display_name) > 8:
                display_name = display_name[:8] + "..."
            parts.append(display_name)
            if slot.is_item:
                parts.append("item")
            elif slot.is_summon:
                parts.append("summon")
            if slot.cost is not None:
                parts.append(f"c:{slot.cost}")
            if slot.quantity is not None:
                parts.append(f"q:{slot.quantity}")
            slot_labels[idx] = "\n".join(parts)
        overlay = self._debug_visualize_bar_rois(
            bar_img,
            total_slots,
            highlight_slot=highlight_slot,
            slot_labels=slot_labels,
            active_slot=highlight_slot,
        )
        self._save_roi_debug(overlay, f"deploy_{int(time.time() * 1000) % 100000000:08d}_state")

    def _process_retreat_forward(self, raw: RawAction):
        """处理 RETREAT：反查格子得到单位，并直接更新回部署栏。"""
        grid = raw.grid
        name = self._deployed.pop(grid, "__unknown__") if grid else "__unknown__"
        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.RETREAT,
                operator_name=name,
                grid=grid,
            )
        )

        if name and not name.startswith("__") and name in self._operators:
            slot = self._operator_info.get(name)
            if slot is None:
                slot = _SlotState(name=name)
                self._log(f"RETREAT 干员 {name} @ {grid}，未找到费用信息")
            self._insert_operator_by_cost(slot)
            self._log(f"RETREAT 干员 {name} @ {grid}，已回到部署栏")
        else:
            self._log(f"RETREAT {name} @ {grid}")

    def _sort_bar_state(self):
        """按费用对 _prev_bar_state 排序，同费用保持现有相对顺序。

        视觉从左到右：低费用干员/召唤物 → 高费用干员/召唤物 → 助战干员 → 道具。
        内部列表为从右到左（index 0 对应最右侧）。
        """
        items = [s for s in self._prev_bar_state if s.is_item or s.is_infinite]
        non_items = [s for s in self._prev_bar_state if not s.is_item and not s.is_infinite]

        support_names = self._support_operator_names()

        def _cost(slot: _SlotState) -> int:
            if slot.cost is not None:
                return slot.cost
            if slot.is_summon:
                return self._summon_costs.get(slot.name, 0)
            if slot.name in self._operator_info:
                return self._operator_info[slot.name].cost or 0
            if slot.name in self._operators:
                return self._operators.index(slot.name)
            return 9999

        def sort_key(slot: _SlotState):
            name = slot.name
            if name in support_names:
                # 助战干员放在道具左侧、高费用干员右侧（从右到左列表中紧接道具之后）
                return (-float("inf"),)
            # 高费用在右（从右到左列表中靠前），低费用在左；
            # Python sort 是稳定的，同费用自然保持输入顺序，避免按 _operators.index 乱序
            return (-_cost(slot),)

        non_items.sort(key=sort_key)
        self._prev_bar_state = items + non_items

    def _insert_operator_by_cost(self, slot: _SlotState):
        """按费用将干员插回 _prev_bar_state 并重新排序。"""
        self._prev_bar_state.append(slot)
        self._sort_bar_state()
        self._log(
            f"干员 {slot.name} 按费用 {slot.cost} 插回部署栏，"
            f"当前 bar={[s.name for s in self._prev_bar_state]}"
        )

    def _process_skill_forward(self, raw: RawAction):
        """处理 SKILL：仅反查格子名称。"""
        grid = raw.grid
        name = self._deployed.get(grid, "__unknown__") if grid else "__unknown__"
        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.SKILL,
                operator_name=name,
                grid=grid,
            )
        )
        self._log(f"SKILL {name} @ {grid}")

    # ------------------------------------------------------------------
    # 初始部署区处理（TEAM_BAR）
    # ------------------------------------------------------------------
    def _process_initial_team_bar(self):
        """处理费用条启动后截取的初始部署区，提前识别每个 slot 的干员/道具/费用/数量。"""
        team_bar_kf = next(
            (kf for kf in self.raw.keyframes.values() if kf.type == KeyframeType.TEAM_BAR),
            None,
        )
        if team_bar_kf is None:
            self._log("未找到 TEAM_BAR 关键帧，跳过初始部署区处理")
            return
        bar_img = self._load_keyframe_image(team_bar_kf.id)
        if bar_img is None:
            self._log("TEAM_BAR 关键帧图片加载失败")
            return

        total_slots = self.raw.initial_operator_count + self.raw.initial_item_count
        if total_slots <= 0:
            self._log("初始干员/道具数为 0，跳过初始部署区处理")
            return

        self._log(f"开始处理初始部署区: total_slots={total_slots}")
        for bar_index in range(total_slots):
            slot_info = self._process_initial_team_bar_slot(bar_img, bar_index, total_slots)
            self._initial_slot_info[bar_index] = slot_info
            if slot_info.is_item:
                self._ensure_item(slot_info.name or self._item_name_for_bar_index(bar_index))
                self._item_bar_index[slot_info.name or self._item_name_for_bar_index(bar_index)] = bar_index
                self._log(
                    f"初始部署区 slot={bar_index}: 道具 {slot_info.name}, quantity={slot_info.quantity}"
                )
            elif slot_info.name and not slot_info.is_unknown:
                self._ensure_operator(slot_info.name)
                self._log(
                    f"初始部署区 slot={bar_index}: 干员 {slot_info.name}, cost={slot_info.cost}"
                )
            else:
                self._log(
                    f"初始部署区 slot={bar_index}: unknown/summon, cost={slot_info.cost}, quantity={slot_info.quantity}"
                )

        if self.debug:
            slot_labels = {}
            for idx, info in self._initial_slot_info.items():
                parts = []
                if info.name:
                    display_name = info.name
                    if len(display_name) > 8:
                        display_name = display_name[:8] + "..."
                    parts.append(display_name)
                if info.cost is not None:
                    parts.append(f"c:{info.cost}")
                if info.quantity is not None:
                    parts.append(f"q:{info.quantity}")
                if parts:
                    slot_labels[idx] = "\n".join(parts)
            overlay = self._debug_visualize_bar_rois(
                bar_img, total_slots, slot_labels=slot_labels
            )
            self._save_roi_debug(overlay, "team_bar_roi_overlay")

    def _process_initial_team_bar_slot(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
    ) -> _InitialSlotInfo:
        """识别初始部署区单个 slot。"""
        slot_info = _InitialSlotInfo(bar_index=bar_index)

        # 头像匹配
        crop = self._crop_slot_from_bar(bar_img, bar_index, total_slots)
        if crop is not None and crop.size > 0:
            name, score = self._match_image(crop)
            if name is not None and score >= self._match_threshold:
                slot_info.name = name
                slot_info.score = score
            else:
                slot_info.is_unknown = True
                if crop is not None:
                    name = self._next_unknown_name()
                    self._templates[name] = crop
                    slot_info.name = name

        # OCR 费用与数量
        if self.ocr is not None:
            slot_info.cost = self._recognize_cost_from_bar_img(bar_img, bar_index, total_slots)
            # 数量 OCR 仅对道具/召唤物；已明确识别为干员时跳过，避免误读费用为数量
            is_operator = (
                slot_info.name is not None
                and not slot_info.is_unknown
                and not slot_info.name.startswith("__")
            )
            if is_operator:
                self._log(f"  初始部署区 quantity slot={bar_index}: 识别为干员，跳过数量 OCR")
            else:
                slot_info.quantity = self._recognize_quantity_from_bar_img(
                    bar_img, bar_index, total_slots
                )

        has_cost = slot_info.cost is not None and slot_info.cost > 0
        has_qty = slot_info.quantity is not None and slot_info.quantity > 0

        # 判定：无费用 + 有数量 → 道具；有费用 → 干员/召唤物；否则按 hint
        if not has_cost and has_qty:
            slot_info.is_item = True
            slot_info.name = self._item_name_for_bar_index(bar_index)
        elif has_cost:
            slot_info.is_item = False
        else:
            slot_info.is_item = 0 <= bar_index < self._item_count_hint
            if slot_info.is_item:
                slot_info.name = self._item_name_for_bar_index(bar_index)

        return slot_info

    def _recognize_cost_from_bar_img(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
    ) -> Optional[int]:
        """从给定的整栏图片中识别指定 slot 的费用。优先使用 ONNX 数字模型。"""
        cost_crop = self._crop_cost_from_bar(bar_img, bar_index, total_slots)
        if cost_crop is None or cost_crop.size == 0:
            self._log(f"  初始部署区 cost slot={bar_index}: ROI 裁剪为空")
            return None

        result = None
        # 1) 优先走 ONNX 数字模型（黑字白底）
        try:
            proc_inv = cost_recognition.preprocess_cost_image_inv(cost_crop)
            t0 = time.perf_counter()
            model_result = self._digit_recognizer.predict_cost(proc_inv)
            model_ms = (time.perf_counter() - t0) * 1000
            if model_result:
                value, conf = model_result
                if 0 <= value <= 99:
                    self._log(
                        f"  初始部署区 cost slot={bar_index}: 模型={value} "
                        f"(conf={conf:.2f}, 耗时={model_ms:.2f}ms)"
                    )
                    result = (value, conf)
        except Exception as e:
            self._log(f"  初始部署区 cost slot={bar_index} 模型异常: {e}")

        # 2) 模型失败则 fallback 到 OCR 双路
        if result is None:
            for preprocess_name, preprocess in (
                ("固定阈值", cost_recognition.preprocess_cost_image),
                ("反色", cost_recognition.preprocess_cost_image_inv),
            ):
                try:
                    proc = preprocess(cost_crop)
                    result = cost_recognition.extract_cost_with_conf(
                        self.ocr.recognize(proc, min_confidence=0.5), min_conf=0.5
                    )
                    if result:
                        self._log(
                            f"  初始部署区 cost slot={bar_index}: {result[0]} "
                            f"(conf={result[1]:.2f}, 方式={preprocess_name})"
                        )
                        break
                    else:
                        self._log(f"  初始部署区 cost slot={bar_index}: 方式={preprocess_name} 未识别到数字")
                except Exception as e:
                    self._log(f"  初始部署区 cost slot={bar_index} 异常 ({preprocess_name}): {e}")
                    continue

        if self.debug:
            self._save_ocr_debug(
                cost_crop,
                f"team_cost_slot_{bar_index}",
                value=result[0] if result else None,
                conf=result[1] if result else None,
            )

        return result[0] if result else None

    def _recognize_quantity_from_bar_img(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int] = None,
    ) -> Optional[int]:
        """从给定的整栏图片中识别指定 slot 头像右下角的数量。优先使用 ONNX 数字模型。"""
        # total_slots <= 12 时，active slot 的数量角标位置整体偏高 _ACTIVE_SELF_Y_SHIFT_PX 像素
        y_shift = -self._ACTIVE_SELF_Y_SHIFT_PX if (
            active_slot is not None
            and bar_index == active_slot
            and total_slots <= 12
        ) else 0
        qty_crop = self._crop_quantity_from_bar(
            bar_img, bar_index, total_slots, active_slot=active_slot, y_shift=y_shift
        )
        if qty_crop is None or qty_crop.size == 0:
            self._log(f"  初始部署区 quantity slot={bar_index}: 数量 ROI 裁剪为空")
            return None

        result = None
        raw_text = ""
        proc = None
        try:
            proc = self._preprocess_quantity_strip(
                cv2.resize(qty_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                invert=True,
            )
        except Exception as e:
            self._log(f"  初始部署区 quantity slot={bar_index} 预处理异常: {e}")
            proc = None

        # 1) 优先走 ONNX 数字模型
        if proc is not None:
            if self.debug:
                self._quantity_slot_debug_counter += 1
                self._save_ocr_debug(
                    proc,
                    f"qty_initial_onnx_input_{bar_index}_{self._quantity_slot_debug_counter:04d}",
                )
            try:
                t0 = time.perf_counter()
                model_result = self._digit_recognizer.predict_quantity(proc)
                model_ms = (time.perf_counter() - t0) * 1000
                if model_result:
                    value, conf = model_result
                    if 0 <= value <= 30:
                        result = (value, conf)
                        self._log(
                            f"  初始部署区 quantity slot={bar_index}: 模型={value} "
                            f"(conf={conf:.2f}, 耗时={model_ms:.2f}ms)"
                        )
            except Exception as e:
                self._log(f"  初始部署区 quantity slot={bar_index} 模型异常: {e}")

        # 2) 模型失败则 fallback 到 OCR
        if result is None and proc is not None and self.ocr is not None:
            try:
                lines = self.ocr.recognize(proc, min_confidence=0.5)
                extracted = self._extract_quantity_from_ocr(lines, min_conf=0.5)
                if extracted:
                    result = (extracted[0], extracted[1])
                    raw_text = extracted[2]
                    self._log(
                        f"  初始部署区 quantity slot={bar_index}: {result[0]} "
                        f"(conf={result[1]:.2f}, 原始='{raw_text}')"
                    )
            except Exception as e:
                self._log(f"  初始部署区 quantity slot={bar_index} OCR 异常: {e}")

        if self.debug:
            self._quantity_slot_debug_counter += 1
            self._save_ocr_debug(
                qty_crop,
                f"team_qty_slot_{bar_index}_{self._quantity_slot_debug_counter:04d}",
                value=result[0] if result else None,
                conf=result[1] if result else None,
            )

        return result[0] if result else None

    def _preclassify_deploys(self):
        """对所有 DEPLOY 动作做一次预分类，记录名称、unknown 状态、最佳槽位数、OCR 费用/数量。"""
        for raw in sorted(self.raw.actions, key=lambda a: a.time_ms):
            if raw.action != ActionType.DEPLOY:
                continue
            info = self._classify_deploy(raw)
            self._deploy_infos[raw.time_ms] = info
            if info.is_item:
                self._item_usage_counts[info.name] = (
                    self._item_usage_counts.get(info.name, 0) + 1
                )

    def _classify_deploy(self, raw: RawAction) -> _DeployInfo:
        bar_index = self._parse_bar_index(raw.target_ref)
        hint_item = bar_index is not None and 0 <= bar_index < self._item_count_hint
        initial = self._initial_slot_info.get(bar_index) if bar_index is not None else None

        # 先做一次头像匹配，得到 best_total 与候选名称，用于后续 OCR 裁剪
        name, score, best_total, best_crop = self._match_or_create_unknown_from_bar(raw, bar_index)
        is_unknown = name.startswith("__unknown_")

        # 尝试用 OCR 判断该 slot 是道具/干员/召唤物
        ocr_cost = self._recognize_cost(raw, bar_index, best_total, active_slot=bar_index) if self.ocr else None
        # 数量 OCR 仅对道具/召唤物；已明确识别为干员时跳过
        is_confident_operator = (
            name is not None
            and not name.startswith("__")
            and not is_unknown
            and score >= self._match_threshold
        ) or (
            initial is not None
            and initial.name is not None
            and not initial.is_item
            and not initial.is_unknown
        )
        if is_confident_operator:
            self._log(f"DEPLOY slot={bar_index} 识别为干员，跳过数量 OCR")
            ocr_qty = None
        else:
            ocr_qty = self._recognize_quantity(raw, bar_index, best_total, active_slot=bar_index) if self.ocr else None
        has_cost = ocr_cost is not None and ocr_cost > 0
        has_qty = ocr_qty is not None and ocr_qty > 0

        # OCR 决策：无费用 + 有数量 → 道具；有费用 → 非道具；否则回退到 hint/初始部署区
        if not has_cost and has_qty:
            is_item = True
            self._log(f"OCR 判定为道具 slot={bar_index}, quantity={ocr_qty}")
        elif has_cost:
            is_item = False
            self._log(f"OCR 判定为非道具 slot={bar_index}, cost={ocr_cost}")
        elif initial is not None:
            is_item = initial.is_item
            self._log(f"OCR 未识别，使用初始部署区判定 slot={bar_index}, is_item={is_item}")
        else:
            is_item = hint_item
            self._log(f"OCR 未识别且缺少初始部署区 slot={bar_index}, 回退到 hint_item={hint_item}")

        if is_item:
            # 若头像匹配阶段误创建了 unknown 模板，清理掉，避免污染模板库
            if name.startswith("__unknown_"):
                self._templates.pop(name, None)
            # 优先使用初始部署区识别的道具名/数量
            if initial is not None and initial.is_item and initial.name:
                name = initial.name
            else:
                name = self._item_name_for_bar_index(bar_index)
            score = 1.0
            is_unknown = False
            best_crop = None
            # 数量优先 OCR，其次初始部署区，最后 None（等待 Pass 3）
            quantity = ocr_qty if has_qty else initial.quantity if initial else None
            self._ensure_item(name)
            self._item_bar_index[name] = bar_index
            self._log(f"预分类 道具 slot={bar_index}, name={name}, quantity={quantity}")
        else:
            # 非道具：若初始部署区已识别出干员名，且本次匹配是 unknown 或低分，优先采用初始部署区名称
            if (
                initial is not None
                and initial.name
                and not initial.is_item
                and not initial.is_unknown
            ):
                if is_unknown or score < self._match_threshold:
                    if is_unknown and name.startswith("__unknown_"):
                        self._templates.pop(name, None)
                    name = initial.name
                    is_unknown = False
                    score = initial.score
                    self._log(f"预分类 干员 slot={bar_index}, 使用初始部署区名称 {name}")
            if is_unknown:
                if best_crop is not None:
                    self._templates[name] = best_crop
                self._log(f"预分类 unknown slot={bar_index}, name={name}, best_total={best_total}")
            else:
                self._ensure_operator(name)
                self._log(f"预分类 干员 slot={bar_index}, name={name}, score={score:.3f}")

        return _DeployInfo(
            raw=raw,
            bar_index=bar_index if bar_index is not None else 0,
            name=name,
            score=score,
            is_item=is_item,
            is_unknown=is_unknown,
            best_total=best_total,
            true_total=None,
            best_crop=best_crop,
            cost=ocr_cost,
            quantity=quantity if is_item else (ocr_qty if has_qty else None),
        )

    # ------------------------------------------------------------------
    # 动作构建阶段（Pass 2）
    # ------------------------------------------------------------------
    def _build_actions(self):
        """根据预分类结果和 predicted_total 生成最终动作序列，并处理 ADD_SUMMON。"""
        # 以 OCR/Hint 中检测到的道具数量为初始道具数，避免 hint 错误导致 predicted_total 偏差
        detected_item_names = {
            info.name for info in self._deploy_infos.values() if info.is_item
        }
        actual_initial_item_count = max(
            self.raw.initial_item_count,
            len(detected_item_names),
        )
        predicted_total = [self.raw.initial_operator_count + actual_initial_item_count]
        item_charges: Dict[str, int] = {}
        summon_charges: Dict[str, int] = {}

        self._log(
            f"构建动作: initial_operator={self.raw.initial_operator_count}, "
            f"initial_item_hint={self.raw.initial_item_count}, "
            f"actual_initial_item={actual_initial_item_count}, "
            f"predicted_total_start={predicted_total[0]}"
        )

        for raw in sorted(self.raw.actions, key=lambda a: a.time_ms):
            if raw.action == ActionType.DEPLOY:
                info = self._deploy_infos.get(raw.time_ms)
                if info is None:
                    self._log(f"警告: DEPLOY @ {raw.time_ms} 缺少预分类信息，跳过")
                    continue

                if info.is_item:
                    self._handle_item_deploy(info, item_charges, predicted_total)
                    continue

                self._handle_unit_deploy(info, summon_charges, predicted_total)

            elif raw.action == ActionType.RETREAT:
                self._handle_retreat(raw, predicted_total)

            elif raw.action == ActionType.SKILL:
                self._handle_skill(raw)

            else:
                self._log(f"忽略未支持的原始动作: {raw.action}")

    def _handle_item_deploy(
        self,
        info: _DeployInfo,
        item_charges: Dict[str, int],
        predicted_total: List[int],
    ):
        """处理道具部署。predicted_total 用单元素列表模拟可变引用。"""
        info.true_total = predicted_total[0]
        if info.name not in item_charges:
            # 优先使用初始部署区/Pass1 OCR 得到的数量，其次用使用次数兜底
            initial_qty = info.quantity
            usage_qty = self._item_usage_counts.get(info.name, 1)
            qty = max(1, initial_qty if initial_qty is not None else usage_qty)
            item_charges[info.name] = qty
            self._log(f"道具 {info.name} 初始数量={qty} (initial_qty={initial_qty}, usage={usage_qty})")

        charges = item_charges[info.name]
        if charges <= 0:
            self._log(f"警告: 道具 {info.name} 数量已耗尽仍尝试部署")
            charges = 1
            item_charges[info.name] = charges

        if charges == 1:
            predicted_total[0] -= 1
            self._log(f"道具 {info.name} 最后一次部署，predicted_total -> {predicted_total[0]}")
        item_charges[info.name] -= 1

        self._actions.append(
            OperatorAction(
                time_ms=info.raw.time_ms,
                action=ActionType.DEPLOY,
                operator_name=info.name,
                grid=info.raw.grid,
                direction=info.raw.direction,
                is_object=True,
            )
        )
        if info.raw.grid:
            self._deployed[info.raw.grid] = info.name

    def _handle_unit_deploy(
        self,
        info: _DeployInfo,
        summon_charges: Dict[str, int],
        predicted_total: List[int],
    ):
        """处理干员/召唤物部署。"""
        actual_total = info.best_total
        info.true_total = predicted_total[0]

        if actual_total > predicted_total[0]:
            diff = actual_total - predicted_total[0]
            if info.is_unknown:
                if info.raw.time_ms not in self._moved_add_summon_times:
                    # 本次 deploy 即为召唤物部署，ADD_SUMMON 应发生在本次部署前
                    qty = info.quantity if info.quantity is not None and info.quantity > 0 else diff
                    charges = max(diff, qty)
                    self._actions.append(
                        OperatorAction(
                            time_ms=info.raw.time_ms,
                            action=ActionType.ADD_SUMMON,
                            operator_name=info.name,
                            grid=(charges, 0),
                        )
                    )
                    self._add_summon_inferences.append(
                        _AddSummonInference(
                            target_name=info.name,
                            action_time_ms=info.raw.time_ms,
                            screenshot_time_ms=info.raw.time_ms,
                            screenshot_bar_index=info.bar_index,
                            screenshot_total=info.best_total,
                            predicted_diff=diff,
                        )
                    )
                    summon_charges[info.name] = summon_charges.get(info.name, 0) + charges
                    predicted_total[0] += charges
                    self._log(
                        f"ADD_SUMMON {info.name} x{charges} 插入到本次部署前 "
                        f"(actual={actual_total}, predicted={predicted_total[0] - charges}, diff={diff})"
                    )
            else:
                # 当前是干员部署，召唤物实际在更早时机已获取，将后续第一个 unknown 的 ADD_SUMMON 前移
                next_unknown = self._find_next_unknown_deploy(info.raw.time_ms)
                if next_unknown is not None:
                    qty = (
                        next_unknown.quantity
                        if next_unknown.quantity is not None and next_unknown.quantity > 0
                        else diff
                    )
                    charges = max(diff, qty)
                    self._actions.append(
                        OperatorAction(
                            time_ms=info.raw.time_ms,
                            action=ActionType.ADD_SUMMON,
                            operator_name=next_unknown.name,
                            grid=(charges, 0),
                        )
                    )
                    self._add_summon_inferences.append(
                        _AddSummonInference(
                            target_name=next_unknown.name,
                            action_time_ms=info.raw.time_ms,
                            screenshot_time_ms=info.raw.time_ms,
                            screenshot_bar_index=next_unknown.bar_index,
                            screenshot_total=info.best_total,
                            predicted_diff=diff,
                        )
                    )
                    summon_charges[next_unknown.name] = (
                        summon_charges.get(next_unknown.name, 0) + charges
                    )
                    predicted_total[0] += charges
                    self._moved_add_summon_times.add(next_unknown.raw.time_ms)
                    self._log(
                        f"ADD_SUMMON {next_unknown.name} x{charges} 前移到干员部署前 "
                        f"@ {info.raw.time_ms} (actual={actual_total}, predicted={predicted_total[0] - charges}, diff={diff})"
                    )
                else:
                    self._log(
                        f"警告: actual={actual_total} > predicted={predicted_total[0]} 但找不到后续 unknown 部署"
                    )

        # 处理部署本身
        if info.is_unknown:
            if info.name not in summon_charges:
                # 未检测到 discrepancy 时的兜底：本次部署前自动补充 1 个
                self._actions.append(
                    OperatorAction(
                        time_ms=info.raw.time_ms,
                        action=ActionType.ADD_SUMMON,
                        operator_name=info.name,
                        grid=(1, 0),
                    )
                )
                self._add_summon_inferences.append(
                    _AddSummonInference(
                        target_name=info.name,
                        action_time_ms=info.raw.time_ms,
                        screenshot_time_ms=info.raw.time_ms,
                        screenshot_bar_index=info.bar_index,
                        screenshot_total=info.best_total,
                        predicted_diff=1,
                    )
                )
                summon_charges[info.name] = 1
                predicted_total[0] += 1
                self._log(f"ADD_SUMMON {info.name} x1 兜底插入到本次部署前")

            charges = summon_charges[info.name]
            if charges == 1:
                predicted_total[0] -= 1
            summon_charges[info.name] -= 1
            if summon_charges[info.name] <= 0:
                del summon_charges[info.name]
            self._summon_deploy_counts[info.name] = (
                self._summon_deploy_counts.get(info.name, 0) + 1
            )
            self._log(
                f"DEPLOY unknown/summon {info.name} @ {info.raw.grid}, "
                f"predicted_total -> {predicted_total[0]}"
            )
        else:
            predicted_total[0] -= 1
            self._log(f"DEPLOY operator {info.name} @ {info.raw.grid}, predicted_total -> {predicted_total[0]}")

        self._actions.append(
            OperatorAction(
                time_ms=info.raw.time_ms,
                action=ActionType.DEPLOY,
                operator_name=info.name,
                grid=info.raw.grid,
                direction=info.raw.direction,
                is_object=False,
            )
        )
        if info.raw.grid:
            self._deployed[info.raw.grid] = info.name

    def _handle_retreat(self, raw: RawAction, predicted_total: List[int]):
        grid = raw.grid
        name = self._deployed.pop(grid, "__unknown__") if grid else "__unknown__"
        # 仅干员撤退会返还部署栏；召唤物撤退不返还，道具不能撤退
        if name and not name.startswith("__") and name in self._operators:
            predicted_total[0] += 1
            self._log(f"RETREAT operator {name} @ {grid}, predicted_total -> {predicted_total[0]}")
        else:
            self._log(f"RETREAT {name} @ {grid}")

        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.RETREAT,
                operator_name=name,
                grid=grid,
            )
        )

    def _handle_skill(self, raw: RawAction):
        grid = raw.grid
        name = self._deployed.get(grid, "__unknown__") if grid else "__unknown__"
        self._actions.append(
            OperatorAction(
                time_ms=raw.time_ms,
                action=ActionType.SKILL,
                operator_name=name,
                grid=grid,
            )
        )
        self._log(f"SKILL {name} @ {grid}")

    def _find_next_unknown_deploy(self, after_time_ms: int) -> Optional[_DeployInfo]:
        """查找 after_time_ms 之后的第一个 unknown DEPLOY。"""
        candidates = [
            info
            for info in self._deploy_infos.values()
            if info.is_unknown and info.raw.time_ms > after_time_ms
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda i: i.raw.time_ms)
        return candidates[0]

    # ------------------------------------------------------------------
    # ADD_SUMMON 反推完成后的 OCR 修正
    # ------------------------------------------------------------------
    def _post_ocr_and_update(self):
        """ADD_SUMMON 反推完成后，利用真实 predicted_total 对召唤物/道具做 OCR 修正。"""
        if self.ocr is None:
            self._log("OCR 引擎不可用，跳过 post-OCR 修正")
            return

        # 1. 道具数量：用本次部署时的真实总槽位数裁剪 ROI
        for info in sorted(self._deploy_infos.values(), key=lambda i: i.raw.time_ms):
            if not info.is_item:
                continue
            if info.true_total is None:
                continue
            qty = self._recognize_item_quantity(
                info.raw, info.bar_index, info.true_total, active_slot=info.bar_index
            )
            if qty is not None and qty > 0:
                info.quantity = qty
                self._log(f"道具 {info.name} OCR 数量={qty}")

        # 2. 召唤物费用与数量
        for info in sorted(self._deploy_infos.values(), key=lambda i: i.raw.time_ms):
            if not info.is_unknown:
                continue
            if info.true_total is None:
                continue
            cost = self._recognize_cost(info.raw, info.bar_index, info.true_total, active_slot=info.bar_index)
            if cost is not None:
                info.cost = cost
            qty = self._recognize_quantity(
                info.raw, info.bar_index, info.true_total, active_slot=info.bar_index
            )
            if qty is not None and qty > 0:
                info.quantity = qty
            self._log(f"召唤物 {info.name} OCR cost={cost}, quantity={qty}")

        # 3. 根据 OCR 数量修正 ADD_SUMMON 动作的 charges
        for inference in self._add_summon_inferences:
            screenshot_info = self._deploy_infos.get(inference.screenshot_time_ms)
            if screenshot_info is None:
                continue
            qty = self._recognize_quantity(
                screenshot_info.raw,
                inference.screenshot_bar_index,
                inference.screenshot_total,
                active_slot=inference.screenshot_bar_index,
            )
            if qty is None or qty <= 0:
                continue
            for action in self._actions:
                if (
                    action.action == ActionType.ADD_SUMMON
                    and action.operator_name == inference.target_name
                    and action.time_ms == inference.action_time_ms
                ):
                    action.grid = (qty, 0)
                    self._log(
                        f"ADD_SUMMON {inference.target_name} charges 修正为 {qty} "
                        f"(预测差值={inference.predicted_diff})"
                    )
                    break

    # ------------------------------------------------------------------
    # 头像匹配
    # ------------------------------------------------------------------
    def _match_or_create_unknown_from_bar(
        self,
        raw: RawAction,
        bar_index: int,
    ) -> Tuple[str, float, int, Optional[np.ndarray]]:
        """从整栏关键帧中尝试多种 slot 总数，取匹配分最高的裁剪结果。

        返回 (name, score, best_total, best_crop)。
        """
        bar_img = self._load_bar_image(raw)
        if bar_img is None:
            name = self._next_unknown_name()
            return name, 0.0, 0, None

        initial_total = self.raw.initial_operator_count + self.raw.initial_item_count
        estimated_total = self._estimate_bar_total_for_matching(raw.time_ms)
        candidates: set = {12}
        if estimated_total < 12:
            candidates.update({13, 14})
        else:
            candidates.update(estimated_total + d for d in range(0, 3))
        candidates = {
            n for n in candidates
            if n > 0 and n >= max(1, self._item_count_hint)
        }
        if not candidates:
            candidates = {max(12, estimated_total)}

        click_offset = self._bar_click_offset_from_right(bar_index, initial_total, bar_img)
        self._log(
            f"DEPLOY slot={bar_index}, 初始总数={initial_total}, "
            f"估算当前总数={estimated_total}, 点击位置距右边缘={click_offset:.1f}px, "
            f"候选总数={sorted(candidates)}"
        )

        best_name: Optional[str] = None
        best_score = -1.0
        best_crop: Optional[np.ndarray] = None
        best_total: Optional[int] = None
        for total in sorted(candidates):
            crop = self._crop_nearest_slot_from_bar(bar_img, click_offset, total)
            if crop is None or crop.size == 0:
                self._log(f"  total={total}: 裁剪失败")
                continue
            nearest_index = self._nearest_slot_index(click_offset, total, bar_img)
            name, score = self._match_image(crop)
            self._log(f"  total={total}, nearest_slot={nearest_index}, match={name}, score={score:.3f}")
            if score > best_score:
                best_score = score
                best_name = name
                best_crop = crop
                best_total = total

        if self.debug and best_total is not None:
            try:
                nearest_index = self._nearest_slot_index(click_offset, best_total, bar_img)
                overlay = self._debug_visualize_bar_rois(
                    bar_img, best_total, highlight_slot=nearest_index
                )
                self._save_roi_debug(
                    overlay, f"deploy_{raw.time_ms:08d}_total{best_total}_slot{bar_index}"
                )
            except Exception as e:
                self._log(f"保存 DEPLOY ROI 调试图失败: {e}")

        if best_name is not None and best_score >= self._match_threshold:
            self._log(f"选择 best_total={best_total}, name={best_name}, score={best_score:.3f}")
            return best_name, best_score, best_total or 0, best_crop

        name = self._next_unknown_name()
        if best_crop is not None:
            self._templates[name] = best_crop
        self._log(
            f"未超过阈值 {self._match_threshold}, 生成占位 {name}, "
            f"best_candidate_total={best_total}, best_score={best_score:.3f}"
        )
        return name, best_score, best_total or 0, best_crop

    def _rough_estimate_bar_total(self, time_ms: int) -> int:
        """保守估算当前总槽位数：所有 DEPLOY 均视为 -1，RETREAT 视为 +1。

        仅用于候选集生成，不替代 Pass 2 中的精确 predicted_total。
        """
        initial_total = self.raw.initial_operator_count + self.raw.initial_item_count
        delta = 0
        for action in sorted(self.raw.actions, key=lambda a: a.time_ms):
            if action.time_ms >= time_ms:
                break
            if action.action == ActionType.DEPLOY:
                delta -= 1
            elif action.action == ActionType.RETREAT:
                delta += 1
        return max(1, initial_total + delta)

    def _estimate_bar_total_for_matching(self, time_ms: int) -> int:
        """用于头像匹配候选集生成的估算：已分类的道具/unknown 不减少总数。

        这里利用预分类阶段已经处理完的前面 DEPLOY 结果：
          - 道具：假设不减少（最后一次使用才减少，此时还不知道 charges）。
          - unknown/summon：不减少（新增单位抵消部署）。
          - 普通干员：-1。
          - 撤退：+1（保守按干员处理）。
        注意：当前正在处理的这次 DEPLOY 不会计入（time_ms >= 才 break）。
        """
        initial_total = self.raw.initial_operator_count + self.raw.initial_item_count
        delta = 0
        # 已分类的前置 DEPLOY
        for info in sorted(self._deploy_infos.values(), key=lambda i: i.raw.time_ms):
            if info.raw.time_ms >= time_ms:
                break
            if info.is_item or info.is_unknown:
                continue
            delta -= 1
        # 前置 RETREAT
        for action in sorted(self.raw.actions, key=lambda a: a.time_ms):
            if action.time_ms >= time_ms:
                break
            if action.action == ActionType.RETREAT:
                delta += 1
        return max(1, initial_total + delta)

    def _load_bar_image(self, raw: RawAction) -> Optional[np.ndarray]:
        for kid in raw.keyframe_ids:
            kf = self.raw.keyframes.get(kid)
            if kf is not None and kf.type == KeyframeType.DEPLOY_BAR:
                return self._load_keyframe_image(kid)
        return None

    def _recover_window_size_from_bar(self, bar_img: np.ndarray) -> Tuple[int, int]:
        """根据整栏关键帧恢复录制时的窗口宽高。"""
        h_bar, w_bar = bar_img.shape[:2]
        window_width = w_bar
        window_height = int(round(h_bar / self._BAR_CAPTURE_HEIGHT_RATIO))
        return window_width, window_height

    def _bar_click_offset_from_right(
        self,
        bar_index: int,
        total_slots: int,
        bar_img: np.ndarray,
    ) -> float:
        """根据录制时假设的总 slot 数，把 bar_index 还原为点击点距右边缘的像素距离。"""
        window_width, _ = self._recover_window_size_from_bar(bar_img)
        cell_w = window_width / 12 if total_slots <= 12 else window_width / total_slots
        return cell_w * (bar_index + 0.5)

    def _crop_nearest_slot_from_bar(
        self,
        bar_img: np.ndarray,
        click_offset_from_right: float,
        total_slots: int,
    ) -> Optional[np.ndarray]:
        """在候选总 slot 数下，找到离物理点击位置最近的 slot 并裁剪其头像。"""
        if total_slots <= 0:
            return None
        nearest_index = self._nearest_slot_index(click_offset_from_right, total_slots, bar_img)
        return self._crop_slot_from_bar(bar_img, nearest_index, total_slots)

    def _nearest_slot_index(
        self,
        click_offset_from_right: float,
        total_slots: int,
        bar_img: np.ndarray,
    ) -> int:
        """在候选总 slot 数下，返回离物理点击位置最近的 slot 索引。"""
        window_width, _ = self._recover_window_size_from_bar(bar_img)
        cell_w = window_width / 12 if total_slots <= 12 else window_width / total_slots
        nearest_index = int(round(click_offset_from_right / cell_w - 0.5))
        return max(0, min(total_slots - 1, nearest_index))

    def _crop_slot_from_bar(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        y_shift: int = 0,
        active_slot: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """假设部署区共有 total_slots 个 slot，从整栏关键帧中裁剪出指定 slot 的头像。

        total_slots > 12 且存在 operator_cost logo 标定时，头像中心 X 采用
        logo_x - 40，否则按动态 cell_w 估算。
        """
        if total_slots <= 0 or bar_index < 0 or bar_index >= total_slots:
            return None

        h_bar, w_bar = bar_img.shape[:2]
        window_width, window_height = self._recover_window_size_from_bar(bar_img)

        cell_w = window_width / 12 if total_slots <= 12 else window_width / total_slots
        bar_center_y = window_height * self._BAR_CENTER_Y_RATIO
        bar_top = window_height * self._BAR_CAPTURE_TOP_RATIO
        cy_rel = bar_center_y - bar_top

        cx = window_width - cell_w * (bar_index + 0.5)
        # total_slots > 12 时优先使用 operator_cost logo 标定修正 X
        if total_slots > 12 and active_slot is not None:
            calibrated = self._get_calibrated_operator_cost_roi(
                total_slots, active_slot, bar_index
            )
            if calibrated is not None:
                logo_x = calibrated[0] * window_width
                cx = logo_x - 40
                self._log(
                    f"  头像 ROI 使用 operator_cost 标定: total={total_slots} "
                    f"active={active_slot} target={bar_index} logo_x={logo_x:.1f}"
                )

        avatar_size = window_height * self._BAR_AVATAR_SIZE_RATIO
        y_offset = avatar_size * self._BAR_AVATAR_Y_OFFSET_RATIO

        crop_cx = int(round(cx))
        crop_cy = int(round(cy_rel + y_offset + y_shift))
        crop_size = int(round(avatar_size))

        x1 = max(0, crop_cx - crop_size // 2)
        y1 = max(0, crop_cy - crop_size // 2)
        x2 = min(w_bar, x1 + crop_size)
        y2 = min(h_bar, y1 + crop_size)

        if x2 - x1 < crop_size * 0.5 or y2 - y1 < crop_size * 0.5:
            return None
        return bar_img[y1:y2, x1:x2]

    def _crop_quantity_from_bar(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int] = None,
        y_shift: int = 0,
    ) -> Optional[np.ndarray]:
        """裁剪指定 slot 右下区域的数量 ROI。

        当 total_slots > 12 且存在标定配置时，使用 (active_slot, bar_index)
        对应的标定 ROI；否则按动态 cell_w 估算。
        """
        if total_slots <= 0 or bar_index < 0 or bar_index >= total_slots:
            return None

        h_bar, w_bar = bar_img.shape[:2]
        window_width, window_height = self._recover_window_size_from_bar(bar_img)
        bar_top = window_height * self._BAR_CAPTURE_TOP_RATIO

        calibrated = None
        if active_slot is not None:
            calibrated = self._get_calibrated_quantity_roi(
                total_slots, active_slot, bar_index
            )

        if calibrated is not None:
            cx_ratio, cy_ratio, half_w_ratio, half_h_ratio = calibrated
            cx = cx_ratio * window_width
            cy_window = cy_ratio * window_height
            half_w = half_w_ratio * window_width
            half_h = half_h_ratio * window_height
            x1 = int(round(cx - half_w))
            x2 = int(round(cx + half_w))
            y1 = int(round(cy_window - bar_top - half_h + y_shift))
            y2 = int(round(cy_window - bar_top + half_h + y_shift))
            self._log(
                f"  数量 ROI 使用标定: total={total_slots} active={active_slot} "
                f"target={bar_index} ({cx_ratio:.4f},{cy_ratio:.4f})"
            )
        else:
            cell_w = (
                window_width / 12 if total_slots <= 12 else window_width / total_slots
            )
            cx = window_width - cell_w * (bar_index + 0.5)
            x1 = int(round(cx))
            x2 = x1 + int(round(cell_w / 2))
            y1 = int(round(window_height * self._QUANTITY_ROI_Y_RATIO - bar_top)) + y_shift
            y2 = y1 + int(round(window_height * self._QUANTITY_ROI_H_RATIO))

        x1 = max(0, x1)
        y1 = max(0, y1)

        # 若使用旧版 180px 高度的截图，底部可能不足，用黑边填充保持尺寸
        pad_right = max(0, x2 - w_bar)
        pad_bottom = max(0, y2 - h_bar)
        if pad_right or pad_bottom:
            bar_img = np.pad(
                bar_img,
                ((0, pad_bottom), (0, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        if x2 <= x1 or y2 <= y1:
            return None
        return bar_img[y1:y2, x1:x2]

    def _match_image(
        self,
        image: Optional[np.ndarray],
        templates: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Optional[str], float]:
        """仅匹配，不创建新模板。返回 (best_name, best_score)，无匹配时 best_name 为 None。

        Args:
            templates: 可选指定一组模板进行匹配，默认使用 self._templates。
        """
        if image is None or image.size == 0:
            return None, 0.0

        templates = templates or self._templates
        if not templates:
            return None, 0.0

        matcher = self._ensure_matcher()
        t0 = time.perf_counter()
        score_matrix = matcher.compute_score_matrix(templates, [image])
        elapsed = (time.perf_counter() - t0) * 1000
        best_name: Optional[str] = None
        best_score = -1.0
        for name, cell_scores in score_matrix.items():
            score = cell_scores.get(0, -1.0)
            if score > best_score:
                best_score = score
                best_name = name

        if self.debug:
            all_scores = sorted(
                ((name, cell_scores.get(0, -1.0)) for name, cell_scores in score_matrix.items()),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            scores_str = ", ".join(f"{n}={s:.3f}" for n, s in all_scores)
            self._log(
                f"  匹配耗时={elapsed:.1f}ms, Top5=[{scores_str}], "
                f"best={best_name}@{best_score:.3f}"
            )

        return best_name, best_score

    def _ensure_matcher(self) -> AvatarMatcherBase:
        if self._matcher is None:
            t0 = time.perf_counter()
            model_name = self._avatar_model_name or os.environ.get(
                "ARK_AUTO_AVATAR_MODEL", "resnet18"
            )
            self._matcher = create_avatar_matcher(
                prefer_resnet=True,
                use_onnx=True,
                input_size=224,
                model_name=model_name,
            )
            providers = getattr(self._matcher, "providers", None)
            providers_str = f", providers={providers}" if providers else ""
            self._log(
                f"头像匹配器初始化完成: model={model_name}, "
                f"耗时={(time.perf_counter() - t0) * 1000:.1f}ms{providers_str}"
            )
        return self._matcher

    def _next_unknown_name(self) -> str:
        self._unknown_counter += 1
        return f"__unknown_{self._unknown_counter}__"

    def _next_summon_obs_name(self) -> str:
        self._summon_obs_counter += 1
        return f"__summon_obs_{self._summon_obs_counter}__"

    def _register_summon_obs(
        self,
        obs_id: str,
        avatar: Optional[np.ndarray],
        cost: Optional[int],
        quantity: Optional[int],
        time_ms: int,
        bar_index: int,
    ) -> _SummonObs:
        obs = self._summon_obs.get(obs_id)
        if obs is None:
            obs = _SummonObs(
                obs_id=obs_id,
                first_seen_time_ms=time_ms,
                first_seen_bar_index=bar_index,
            )
            self._summon_obs[obs_id] = obs
        if avatar is not None:
            obs.avatars.append(avatar)
        if cost is not None:
            obs.costs.append(cost)
        if quantity is not None:
            obs.quantities.append(quantity)
        return obs

    def _cluster_summon_obs(self) -> Dict[str, str]:
        """对所有召唤物观察做基于头像相似度的聚类，返回 obs_id -> final_name 映射。

        每个 obs_id 在解析阶段都是一次独立观察。聚类时取每个 obs 的第一张头像
        作为代表，计算两两相似度，超过阈值则归为同一类，最终命名为
        __unknown_1__、__unknown_2__ 等。
        """
        t0 = time.perf_counter()
        obs_ids = list(self._summon_obs.keys())
        if not obs_ids:
            return {}

        rep_avatars: Dict[str, np.ndarray] = {}
        for obs_id in obs_ids:
            obs = self._summon_obs[obs_id]
            if obs.avatars:
                rep_avatars[obs_id] = obs.avatars[0]

        obs_id_to_final: Dict[str, str] = {}
        if not rep_avatars:
            for obs_id in obs_ids:
                obs_id_to_final[obs_id] = self._next_unknown_name()
            self._log(f"[耗时] 召唤物聚类 (无头像): {(time.perf_counter()-t0)*1000:.1f}ms")
            return obs_id_to_final

        if len(rep_avatars) == 1:
            obs_id_to_final[list(rep_avatars.keys())[0]] = self._next_unknown_name()
            self._log(f"[耗时] 召唤物聚类 (单观察): {(time.perf_counter()-t0)*1000:.1f}ms")
            return obs_id_to_final

        matcher = self._ensure_matcher()
        # 一次性计算所有代表头像之间的相似度矩阵
        avatar_list = [rep_avatars[oid] for oid in obs_ids]
        score_matrix = matcher.compute_score_matrix(rep_avatars, avatar_list)

        n = len(obs_ids)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i, obs_i in enumerate(obs_ids):
            for j, obs_j in enumerate(obs_ids):
                if i >= j:
                    continue
                # score_matrix[obs_i] 是以 obs_i 为模板对 image j 的匹配分
                score_ij = score_matrix.get(obs_i, {}).get(j, -1.0)
                score_ji = score_matrix.get(obs_j, {}).get(i, -1.0)
                score = max(score_ij, score_ji)
                if score >= self._SUMMON_CLUSTER_THRESHOLD:
                    union(i, j)
                    if self.debug:
                        self._log(
                            f"  聚类合并 {obs_i} <-> {obs_j}, score={score:.3f}"
                        )

        clusters: Dict[int, List[str]] = {}
        for idx, obs_id in enumerate(obs_ids):
            root = find(idx)
            clusters.setdefault(root, []).append(obs_id)

        obs_id_to_final: Dict[str, str] = {}
        for members in clusters.values():
            final_name = self._next_unknown_name()
            for obs_id in members:
                obs_id_to_final[obs_id] = final_name

        if self.debug:
            for members in clusters.values():
                final = obs_id_to_final[members[0]]
                self._log(f"  召唤物聚类: {members} -> {final}")

        self._log(f"[耗时] 召唤物聚类 (n={len(obs_ids)}): {(time.perf_counter()-t0)*1000:.1f}ms")
        return obs_id_to_final

    def _rewrite_actions_with_final_summon_names(self):
        """把 actions 中的 obs_id 替换为聚类后的最终召唤物名称，并去重 ADD_SUMMON。

        去重规则基于库存模拟：
          - 库存为 0 时出现的 ADD_SUMMON：保留（首次获取或消耗完再补充）。
          - 库存 > 0 且新观察数量 > 当前库存：保留（中场补充了更多）。
          - 库存 > 0 且新观察数量 <= 当前库存：视为同帧重复观察，删除。
        """
        if not self._obs_id_to_final:
            return

        # 收集所有召唤物相关事件：ADD_SUMMON / DEPLOY
        events: List[Tuple[int, str, str, int]] = []
        for idx, action in enumerate(self._actions):
            final_name = self._obs_id_to_final.get(action.operator_name or "")
            if final_name is None:
                continue
            if action.action == ActionType.ADD_SUMMON:
                qty = (
                    action.grid[0]
                    if action.grid and len(action.grid) > 0 and action.grid[0] > 0
                    else 1
                )
                events.append((idx, final_name, "add", qty))
            elif action.action == ActionType.DEPLOY:
                events.append((idx, final_name, "deploy", 1))

        # 按最终名称模拟库存，决定保留哪些 ADD_SUMMON
        keep_add_summon_indices: Set[int] = set()
        final_names = sorted({e[1] for e in events})
        for final_name in final_names:
            stock = 0
            for idx, fn, etype, value in events:
                if fn != final_name:
                    continue
                if etype == "add":
                    if stock <= 0:
                        keep_add_summon_indices.add(idx)
                        stock = value
                    elif value > stock:
                        # 中场补充，数量变多了
                        keep_add_summon_indices.add(idx)
                        stock = value
                    elif self.debug:
                        self._log(
                            f"  去重 ADD_SUMMON stock={stock} >= value={value} "
                            f"for {final_name} @ {self._actions[idx].time_ms}"
                        )
                elif etype == "deploy":
                    stock = max(0, stock - value)

        # 重写 actions：替换 obs_id 为最终名称，并删除不保留的 ADD_SUMMON
        new_actions: List[OperatorAction] = []
        for idx, action in enumerate(self._actions):
            final_name = self._obs_id_to_final.get(action.operator_name or "")
            if final_name is None:
                new_actions.append(action)
                continue
            if (
                action.action == ActionType.ADD_SUMMON
                and idx not in keep_add_summon_indices
            ):
                continue
            new_actions.append(
                action.model_copy(update={"operator_name": final_name})
            )
        self._actions = new_actions

        # 按最终名称聚合费用和部署次数
        new_costs: Dict[str, int] = {}
        new_counts: Dict[str, int] = {}
        for obs_id, final_name in self._obs_id_to_final.items():
            obs = self._summon_obs.get(obs_id)
            if obs is None:
                continue
            # 费用取出现次数最多的；冲突时优先非零值
            cost = self._summon_costs.get(obs_id, 0)
            if obs.costs:
                cost = max(set(obs.costs), key=obs.costs.count)
            if final_name not in new_costs or (
                new_costs[final_name] == 0 and cost != 0
            ):
                new_costs[final_name] = cost

            count = self._summon_deploy_counts.get(obs_id, 0)
            new_counts[final_name] = new_counts.get(final_name, 0) + count

        # 保留非 obs_id 来源的召唤物条目（如无限道具），避免被聚类重写清空
        for name, cost in self._summon_costs.items():
            if name not in self._obs_id_to_final and name not in new_costs:
                new_costs[name] = cost
        for name, count in self._summon_deploy_counts.items():
            if name not in self._obs_id_to_final and name not in new_counts:
                new_counts[name] = count

        self._summon_costs = new_costs
        self._summon_deploy_counts = new_counts

    def _detect_removed_summons(self):
        """基于 DEPLOY 前后状态快照检测因 RETREAT 消失的召唤物，生成 REMOVE_SUMMON。

        召唤物在解析阶段使用临时 obs_id，聚类后才有最终名称。本方法在聚类/重写
        完成后执行，将快照中的 obs_id 映射到 final name，通过最终名称集合差异
        判断哪些召唤物在两次 DEPLOY 之间从部署栏完全消失。
        """
        if not self._deploy_snapshots:
            return
        t0 = time.perf_counter()
        removed_actions: List[OperatorAction] = []

        for time_ms, prev_state, pre_state, deployed_name in self._deploy_snapshots:
            def _summon_finals(state: List[_SlotState]) -> Set[str]:
                result: Set[str] = set()
                for slot in state:
                    if slot.is_summon and slot.name:
                        final = self._obs_id_to_final.get(slot.name)
                        if final:
                            result.add(final)
                return result

            prev_finals = _summon_finals(prev_state)
            pre_finals = _summon_finals(pre_state)
            deployed_final = self._obs_id_to_final.get(deployed_name or "")

            disappeared = prev_finals - pre_finals - {deployed_final}
            for final_name in sorted(disappeared):
                removed_actions.append(
                    OperatorAction(
                        time_ms=max(0, time_ms - 1),
                        action=ActionType.REMOVE_SUMMON,
                        operator_name=final_name,
                    )
                )
                self._log(f"REMOVE_SUMMON {final_name} @ {max(0, time_ms - 1)}ms (DEPLOY@{time_ms} 前消失)")

        if removed_actions:
            self._actions.extend(removed_actions)
            self._actions.sort(key=lambda a: a.time_ms)
        self._log(f"[耗时] 检测移除的召唤物: {(time.perf_counter() - t0) * 1000:.1f}ms, 生成 {len(removed_actions)} 个 REMOVE_SUMMON")

    def _compute_red_ratio(self, avatar: np.ndarray) -> float:
        """计算头像 ROI 内红像素占比，用于识别冷却中的召唤物。

        使用 HSV 色相检测，对暗红/酒红色冷却背景比 RGB 阈值更鲁棒。
        """
        if avatar is None or avatar.size == 0:
            return 0.0
        if len(avatar.shape) == 2:
            return 0.0
        if avatar.shape[2] == 4:
            avatar = cv2.cvtColor(avatar, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(avatar, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 红色色相在 0° 附近环绕：0~10 与 170~180 都视为红
        red_mask = (
            ((h <= self._RED_COOLDOWN_HUE_LOW) | (h >= self._RED_COOLDOWN_HUE_HIGH))
            & (s > self._RED_COOLDOWN_SAT_MIN)
            & (v > self._RED_COOLDOWN_VAL_MIN)
        )
        return float(np.count_nonzero(red_mask)) / red_mask.size

    def _identify_cooldown_summons(self) -> Dict[str, str]:
        """聚类完成后，对仍为 __unknown_N__ 的聚类检测红像素。

        红像素占比高的 unknown 视为处于冷却状态的已有召唤物，按 OCR 费用匹配到
        现有召唤物；同费用时取部署频率最高者。返回 unknown_name -> existing_name 映射。
        """
        t0 = time.perf_counter()
        relabeled: Dict[str, str] = {}

        # 按最终名称聚合 obs_id
        final_to_obs: Dict[str, List[str]] = {}
        for obs_id, final_name in self._obs_id_to_final.items():
            final_to_obs.setdefault(final_name, []).append(obs_id)

        unknown_finals = [name for name in final_to_obs if name.startswith("__unknown_")]
        if not unknown_finals:
            self._log("[耗时] 识别冷却召唤物: 无 unknown 聚类")
            return relabeled

        # 计算非 unknown 召唤物头像的红像素基线（用于日志调试）
        baseline_avatars: List[np.ndarray] = []
        for final_name, obs_ids in final_to_obs.items():
            if final_name.startswith("__unknown_"):
                continue
            for obs_id in obs_ids:
                obs = self._summon_obs.get(obs_id)
                if obs and obs.avatars:
                    baseline_avatars.append(obs.avatars[0])
                    break

        if baseline_avatars:
            baseline_ratios = [self._compute_red_ratio(a) for a in baseline_avatars]
            baseline_mean = float(np.mean(baseline_ratios))
            baseline_std = float(np.std(baseline_ratios))
        else:
            baseline_mean = 0.0
            baseline_std = 0.0

        # 阈值不低于固定下限，避免正常截图偏红时误检
        threshold = max(
            self._RED_COOLDOWN_MIN_RATIO,
            baseline_mean + 3 * max(baseline_std, 0.02),
        )
        self._log(
            f"红像素基线 mean={baseline_mean:.3f}, std={baseline_std:.3f}, threshold={threshold:.3f}"
        )

        # 现有召唤物名称集合（非 unknown / obs_id）
        existing_summons = {
            name
            for name in self._summon_costs.keys()
            if not name.startswith("__unknown_") and not name.startswith("__summon_obs_")
        }

        for unknown_name in unknown_finals:
            obs_ids = final_to_obs[unknown_name]
            rep_avatar = None
            rep_cost = None
            for obs_id in obs_ids:
                obs = self._summon_obs.get(obs_id)
                if obs is None:
                    continue
                if rep_avatar is None and obs.avatars:
                    rep_avatar = obs.avatars[0]
                if rep_cost is None and obs.costs:
                    rep_cost = max(set(obs.costs), key=obs.costs.count)
                if rep_avatar is not None and rep_cost is not None:
                    break

            if rep_avatar is None:
                continue

            red_ratio = self._compute_red_ratio(rep_avatar)
            self._log(f"unknown {unknown_name}: red_ratio={red_ratio:.3f}, cost={rep_cost}")

            if red_ratio < threshold:
                continue

            if not rep_cost:
                continue

            candidates = [
                name
                for name in existing_summons
                if self._summon_costs.get(name) == rep_cost
            ]
            if not candidates:
                self._log(
                    f"  {unknown_name} 红像素达标但无费用={rep_cost} 的现有召唤物"
                )
                continue

            # 同费用取部署频率最高者
            candidates.sort(
                key=lambda n: self._summon_deploy_counts.get(n, 0), reverse=True
            )
            best_match = candidates[0]
            relabeled[unknown_name] = best_match
            self._log(
                f"  冷却召唤物 {unknown_name} -> {best_match} (cost={rep_cost})"
            )

        if not relabeled:
            self._log(
                f"[耗时] 识别冷却召唤物: 无匹配 ({(time.perf_counter() - t0) * 1000:.1f}ms)"
            )
            return relabeled

        # 更新 _obs_id_to_final
        for obs_id, final_name in list(self._obs_id_to_final.items()):
            if final_name in relabeled:
                self._obs_id_to_final[obs_id] = relabeled[final_name]

        # 重命名 actions
        for i, action in enumerate(self._actions):
            if action.operator_name in relabeled:
                self._actions[i] = action.model_copy(
                    update={"operator_name": relabeled[action.operator_name]}
                )

        # 合并 _summon_costs 与 _summon_deploy_counts
        for unknown_name, existing_name in relabeled.items():
            self._summon_costs.pop(unknown_name, None)
            if unknown_name in self._summon_deploy_counts:
                self._summon_deploy_counts[existing_name] = (
                    self._summon_deploy_counts.get(existing_name, 0)
                    + self._summon_deploy_counts.pop(unknown_name, 0)
                )

        self._log(
            f"[耗时] 识别冷却召唤物: 重命名 {len(relabeled)} 个 "
            f"({(time.perf_counter() - t0) * 1000:.1f}ms)"
        )
        return relabeled

    def _cleanup_redundant_summon_actions(self, relabeled: Dict[str, str]):
        """删除/修正因冷却误识别产生的冗余 ADD_SUMMON / REMOVE_SUMMON 对。

        仅处理本次被重命名的召唤物。规则：
          - ADD → REMOVE 中间无 DEPLOY：直接删除两者（新增Charge未使用即消失）；
          - REMOVE → ADD 中间无 DEPLOY：ADD 数量修正为 (ADD数量 - REMOVE数量)，
            其中 REMOVE 数量从该次 REMOVE 对应的部署栏快照 prev_state 中读取；
            若修正后数量 <= 0 则删除两者；
          - 中间存在 DEPLOY：保留原动作，避免影响部署索引。
        """
        if not relabeled:
            return
        t0 = time.perf_counter()
        target_summons = set(relabeled.values())

        # 从快照中反查每个 REMOVE_SUMMON 动作对应的移除数量
        remove_qty_map: Dict[Tuple[str, int], int] = {}
        for snap_time_ms, prev_state, pre_state, deployed_name in self._deploy_snapshots:
            def _final_name(slot_name: Optional[str]) -> Optional[str]:
                if not slot_name:
                    return None
                return self._obs_id_to_final.get(slot_name, slot_name)

            prev_finals: Dict[str, int] = {}
            for slot in prev_state:
                final = _final_name(slot.name)
                if final:
                    prev_finals[final] = slot.quantity or 1

            pre_finals: Set[str] = set()
            for slot in pre_state:
                final = _final_name(slot.name)
                if final:
                    pre_finals.add(final)

            deployed_final = _final_name(deployed_name)

            for final_name, qty in prev_finals.items():
                if final_name in pre_finals or final_name == deployed_final:
                    continue
                remove_time_ms = max(0, snap_time_ms - 1)
                remove_qty_map[(final_name, remove_time_ms)] = qty

        new_actions: List[OperatorAction] = []
        i = 0
        removed_count = 0
        corrected_count = 0

        while i < len(self._actions):
            action = self._actions[i]
            name = action.operator_name

            # 非目标召唤物或非 ADD/REMOVE：直接保留
            if (
                name not in target_summons
                or action.action
                not in (ActionType.ADD_SUMMON, ActionType.REMOVE_SUMMON)
            ):
                new_actions.append(action)
                i += 1
                continue

            # 寻找同一召唤物的下一个相反动作，且中间无 DEPLOY
            paired_idx = None
            for j in range(i + 1, len(self._actions)):
                mid = self._actions[j]
                if mid.action == ActionType.DEPLOY:
                    break
                if (
                    mid.operator_name == name
                    and mid.action
                    in (ActionType.ADD_SUMMON, ActionType.REMOVE_SUMMON)
                ):
                    if mid.action != action.action:
                        paired_idx = j
                    break

            if paired_idx is None:
                # 无配对，直接保留
                new_actions.append(action)
                i += 1
                continue

            other = self._actions[paired_idx]

            if (
                action.action == ActionType.REMOVE_SUMMON
                and other.action == ActionType.ADD_SUMMON
            ):
                # REMOVE -> ADD：修正 ADD 数量
                add_qty = (
                    other.grid[0]
                    if other.grid and len(other.grid) > 0
                    else 1
                )
                remove_qty = remove_qty_map.get((name, action.time_ms), 1)
                net_qty = add_qty - remove_qty
                if net_qty > 0:
                    corrected = other.model_copy(
                        update={"grid": (net_qty, 0)}
                    )
                    new_actions.append(corrected)
                    corrected_count += 1
                    self._log(
                        f"  修正 ADD_SUMMON {name}@{paired_idx}: {add_qty} - {remove_qty} = {net_qty}"
                    )
                else:
                    removed_count += 2
                    self._log(
                        f"  删除冗余 REMOVE@{i}({remove_qty}) / ADD@{paired_idx}({add_qty}) for {name}"
                    )
                i = paired_idx + 1

            elif (
                action.action == ActionType.ADD_SUMMON
                and other.action == ActionType.REMOVE_SUMMON
            ):
                # ADD -> REMOVE：直接删除两者
                removed_count += 2
                self._log(
                    f"  删除冗余 ADD@{i} / REMOVE@{paired_idx} for {name}"
                )
                i = paired_idx + 1

            else:
                new_actions.append(action)
                i += 1

        if removed_count or corrected_count:
            self._actions = new_actions
        self._log(
            f"[耗时] 清理冗余召唤物动作: 删除 {removed_count} 个, 修正 {corrected_count} 个 "
            f"({(time.perf_counter() - t0) * 1000:.1f}ms)"
        )

    def _prune_redundant_add_summons_by_quantity(self):
        """按召唤物数量时间线修剪冗余 ADD_SUMMON。

        对每个召唤物，收集每次 DEPLOY 前部署栏中的数量。相邻两次观察之间：
          - 若本次数量等于“上次数量 - 上次是否部署了该召唤物”（即没有净增加），
            则期间的 ADD_SUMMON 是冗余的，删除；
          - 否则视为合法补充，保留。
        第一次观察与初始部署栏数量比较。
        """
        if not self._deploy_snapshots:
            return
        t0 = time.perf_counter()

        observations: Dict[str, List[Tuple[int, int, bool]]] = {}
        for snap_time_ms, _prev_state, pre_state, deployed_name in self._deploy_snapshots:
            deployed_final = self._obs_id_to_final.get(deployed_name, deployed_name)
            for slot in pre_state:
                if not slot.is_summon or slot.is_infinite or slot.quantity is None:
                    continue
                final_name = self._obs_id_to_final.get(slot.name, slot.name)
                if not final_name:
                    continue
                is_self_deploy = final_name == deployed_final
                observations.setdefault(final_name, []).append(
                    (snap_time_ms, slot.quantity, is_self_deploy)
                )

        for name in observations:
            observations[name].sort(key=lambda x: x[0])

        indices_to_delete: Set[int] = set()
        for idx, action in enumerate(self._actions):
            if action.action != ActionType.ADD_SUMMON or not action.operator_name:
                continue
            name = action.operator_name
            obs = observations.get(name)
            if not obs:
                continue
            t = action.time_ms
            prev_time: Optional[int] = None
            prev_qty = self._initial_summon_quantities.get(name, 0)
            prev_self_deploy = False
            for d_time, d_qty, is_self in obs:
                expected_qty = prev_qty - (1 if prev_self_deploy else 0)
                if (prev_time is None or t > prev_time) and t <= d_time:
                    if d_qty == expected_qty:
                        indices_to_delete.add(idx)
                        if self.debug:
                            self._log(
                                f"  删除冗余 ADD_SUMMON {name} @ {t}ms "
                                f"(期望数量 {expected_qty} == 实际 {d_qty})"
                            )
                    break
                prev_time = d_time
                prev_qty = d_qty
                prev_self_deploy = is_self

        if indices_to_delete:
            self._actions = [
                a for i, a in enumerate(self._actions)
                if i not in indices_to_delete
            ]
        self._log(
            f"[耗时] 按数量修剪冗余 ADD_SUMMON: 删除 {len(indices_to_delete)} 个 "
            f"({(time.perf_counter() - t0) * 1000:.1f}ms)"
        )

    def _correct_add_summon_deltas(self):
        """全局修正 ADD_SUMMON 数量为真实 delta（新增 Charge 数）。

        遍历 actions，维护每个召唤物的当前持有数量。对每个 ADD_SUMMON：
          真实 delta = 声明数量 - 当前持有数量
          - 若 delta > 0：保留 ADD，数量改为 delta；
          - 若 delta <= 0：删除该 ADD（没有实际新增 Charge）。

        REMOVE_SUMMON 的移除数量从部署栏快照 prev_state 中读取；
        DEPLOY 消耗 1 个 Charge。
        """
        t0 = time.perf_counter()

        # 从快照中反查每个 REMOVE_SUMMON 动作对应的移除数量
        remove_qty_map: Dict[Tuple[str, int], int] = {}
        for snap_time_ms, prev_state, pre_state, deployed_name in self._deploy_snapshots:
            def _final_name(slot_name: Optional[str]) -> Optional[str]:
                if not slot_name:
                    return None
                return self._obs_id_to_final.get(slot_name, slot_name)

            prev_finals: Dict[str, int] = {}
            for slot in prev_state:
                final = _final_name(slot.name)
                if final:
                    prev_finals[final] = slot.quantity or 1

            pre_finals: Set[str] = set()
            for slot in pre_state:
                final = _final_name(slot.name)
                if final:
                    pre_finals.add(final)

            deployed_final = _final_name(deployed_name)

            for final_name, qty in prev_finals.items():
                if final_name in pre_finals or final_name == deployed_final:
                    continue
                remove_time_ms = max(0, snap_time_ms - 1)
                remove_qty_map[(final_name, remove_time_ms)] = qty

        current_qty: Dict[str, int] = dict(self._initial_summon_quantities)
        if self.debug and current_qty:
            self._log(f"  初始召唤物数量作为 delta 修正基准: {current_qty}")
        new_actions: List[OperatorAction] = []
        corrected_count = 0
        deleted_count = 0

        for action in self._actions:
            name = action.operator_name
            if action.action == ActionType.ADD_SUMMON:
                claimed = (
                    action.grid[0]
                    if action.grid and len(action.grid) > 0
                    else 1
                )
                before = current_qty.get(name, 0)
                real_delta = claimed - before
                if real_delta > 0:
                    if real_delta != claimed:
                        corrected = action.model_copy(
                            update={"grid": (real_delta, 0)}
                        )
                        new_actions.append(corrected)
                        corrected_count += 1
                        self._log(
                            f"  修正 ADD_SUMMON {name}: {claimed} - {before} = {real_delta}"
                        )
                    else:
                        new_actions.append(action)
                    current_qty[name] = before + real_delta
                else:
                    deleted_count += 1
                    self._log(
                        f"  删除冗余 ADD_SUMMON {name}: {claimed} <= 当前持有 {before}"
                    )
            elif action.action == ActionType.REMOVE_SUMMON:
                new_actions.append(action)
                remove_qty = remove_qty_map.get((name, action.time_ms), 1)
                current_qty[name] = max(
                    0, current_qty.get(name, 0) - remove_qty
                )
            elif action.action == ActionType.DEPLOY:
                new_actions.append(action)
                if name:
                    current_qty[name] = max(
                        0, current_qty.get(name, 0) - 1
                    )
            else:
                new_actions.append(action)

        self._actions = new_actions
        self._log(
            f"[耗时] 修正 ADD_SUMMON delta: 修正 {corrected_count} 个, 删除 {deleted_count} 个 "
            f"({(time.perf_counter() - t0) * 1000:.1f}ms)"
        )

    def _ocr_name_card(self, keyframe_id: str) -> Optional[str]:
        """对 DEPLOY_NAME_CARD 关键帧做黑底白字 OCR，返回识别到的最佳文本。"""
        if self.ocr is None:
            return None
        img = self._load_keyframe_image(keyframe_id)
        if img is None or img.size == 0:
            return None
        try:
            if len(img.shape) == 3 and img.shape[2] in (3, 4):
                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
                )
            else:
                gray = img
            # 黑底白字：反色后按白字识别
            inv = 255 - gray
            lines = self.ocr.recognize(inv, min_confidence=0.5)
            best_text = None
            best_conf = 0.0
            for _bbox, (text, conf) in lines:
                text = text.strip()
                if text and conf > best_conf:
                    best_conf = conf
                    best_text = text
            if best_text:
                self._log(f"名称卡 OCR {keyframe_id}: {best_text} (conf={best_conf:.2f})")
            else:
                self._log(f"名称卡 OCR {keyframe_id}: 未识别到文本")
            if self.debug:
                self._save_ocr_debug(gray, f"namecard_orig_{keyframe_id}")
                self._save_ocr_debug(inv, f"namecard_inv_{keyframe_id}")
            return best_text
        except Exception as e:
            self._log(f"名称卡 OCR {keyframe_id} 异常: {e}")
            return None

    def _rewrite_deploy_card_names(self):
        """利用部署时截取的名称卡关键帧，对道具/召唤物占位符进行 OCR 重命名。"""
        if not self.ocr or not self._name_card_kf_by_name:
            return
        t0 = time.perf_counter()

        # 名称卡按部署时的临时名称收集：
        #   - 道具: __item_N__
        #   - 无限道具: __infinite_item_N__
        #   - 召唤物（聚类前）: __summon_obs_N__，聚类后映射为 __unknown_N__
        #   - 兜底未知: __unknown_N__
        target_to_kf_ids: Dict[str, List[str]] = {}
        for collected_name, kf_ids in self._name_card_kf_by_name.items():
            target_name = None
            if collected_name.startswith("__item_"):
                target_name = collected_name
            elif collected_name.startswith("__infinite_item_"):
                target_name = collected_name
            elif collected_name.startswith("__summon_obs_"):
                target_name = self._obs_id_to_final.get(collected_name)
            elif collected_name.startswith("__unknown_"):
                target_name = collected_name
            if target_name and (
                target_name.startswith("__item_")
                or target_name.startswith("__infinite_item_")
                or target_name.startswith("__unknown_")
            ):
                target_to_kf_ids.setdefault(target_name, []).extend(kf_ids)

        renames: Dict[str, str] = {}
        used_names: Set[str] = set(self._operators)
        used_names.update(it.name for it in self._items)

        for target_name, kf_ids in target_to_kf_ids.items():
            new_name = None
            for kf_id in kf_ids:
                text = self._ocr_name_card(kf_id)
                if text:
                    new_name = text
                    break
            if not new_name:
                continue
            # 避免与已有名称冲突
            base = new_name
            suffix = 1
            while new_name in used_names:
                new_name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(new_name)
            renames[target_name] = new_name
            self._log(f"名称卡重命名: {target_name} -> {new_name}")

        if not renames:
            self._log(f"[耗时] OCR 重命名: 无占位符需识别")
            return

        # 重命名 items
        for it in self._items:
            if it.name in renames:
                it.name = renames[it.name]

        # 重命名 summon 相关字典
        for old_name, new_name in list(renames.items()):
            if old_name in self._summon_costs:
                self._summon_costs[new_name] = self._summon_costs.pop(old_name)
            if old_name in self._summon_deploy_counts:
                self._summon_deploy_counts[new_name] = self._summon_deploy_counts.pop(old_name)

        # 重命名 item 相关字典
        for old_name, new_name in list(renames.items()):
            if old_name in self._item_usage_counts:
                self._item_usage_counts[new_name] = self._item_usage_counts.pop(old_name)
            if old_name in self._item_initial_quantity:
                self._item_initial_quantity[new_name] = self._item_initial_quantity.pop(old_name)
            if old_name in self._item_bar_index:
                self._item_bar_index[new_name] = self._item_bar_index.pop(old_name)

        # 重命名无限道具内部状态
        for item in self._infinite_items.values():
            if item.name in renames:
                item.name = renames[item.name]

        # 重命名 actions 与场上部署映射
        for i, action in enumerate(self._actions):
            if action.operator_name in renames:
                self._actions[i] = action.model_copy(update={"operator_name": renames[action.operator_name]})
        for grid, name in list(self._deployed.items()):
            if name in renames:
                self._deployed[grid] = renames[name]

        # 同步更新 _obs_id_to_final，避免后续红像素检测等步骤仍看到旧的 __unknown_N__
        for old_name, new_name in renames.items():
            if not old_name.startswith("__unknown_"):
                continue
            for obs_id, final_name in list(self._obs_id_to_final.items()):
                if final_name == old_name:
                    self._obs_id_to_final[obs_id] = new_name

        # 同步更新初始召唤物数量表的键名，保证后续数量比较使用的是最终名称
        for old_name, new_name in list(renames.items()):
            if old_name in self._initial_summon_quantities:
                self._initial_summon_quantities[new_name] = self._initial_summon_quantities.pop(old_name)

        self._log(f"[耗时] OCR 重命名: {(time.perf_counter() - t0) * 1000:.1f}ms, 重命名 {len(renames)} 个")

    # ------------------------------------------------------------------
    # OCR 费用与数量
    # ------------------------------------------------------------------
    def _recognize_cost(
        self,
        raw: RawAction,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int] = None,
    ) -> Optional[int]:
        """识别指定 slot 的费用数字。优先使用 ONNX 数字模型。"""
        if self.ocr is None:
            self._log(f"  OCR cost slot={bar_index}: ocr 未初始化，跳过")
            return None
        bar_img = self._load_bar_image(raw)
        if bar_img is None:
            self._log(f"  OCR cost slot={bar_index}: 未找到 DEPLOY_BAR 关键帧")
            return None
        cost_crop = self._crop_cost_from_bar(
            bar_img, bar_index, total_slots, active_slot=active_slot
        )
        if cost_crop is None or cost_crop.size == 0:
            self._log(f"  OCR cost slot={bar_index}: 费用 ROI 裁剪为空")
            return None

        result = None
        # 1) 优先走 ONNX 数字模型（黑字白底）
        try:
            proc_inv = cost_recognition.preprocess_cost_image_inv(cost_crop)
            t0 = time.perf_counter()
            model_result = self._digit_recognizer.predict_cost(proc_inv)
            model_ms = (time.perf_counter() - t0) * 1000
            if model_result:
                value, conf = model_result
                if 0 <= value <= 99:
                    self._log(
                        f"  OCR cost slot={bar_index}: 模型={value} "
                        f"(conf={conf:.2f}, 耗时={model_ms:.2f}ms)"
                    )
                    result = (value, conf)
        except Exception as e:
            self._log(f"  OCR cost slot={bar_index} 模型异常: {e}")

        # 2) 模型失败则 fallback 到 OCR 双路
        if result is None:
            for preprocess_name, preprocess in (
                ("固定阈值", cost_recognition.preprocess_cost_image),
                ("反色", cost_recognition.preprocess_cost_image_inv),
            ):
                try:
                    proc = preprocess(cost_crop)
                    result = cost_recognition.extract_cost_with_conf(
                        self.ocr.recognize(proc, min_confidence=0.5), min_conf=0.5
                    )
                    if result:
                        self._log(
                            f"  OCR cost slot={bar_index}: {result[0]} "
                            f"(conf={result[1]:.2f}, 方式={preprocess_name})"
                        )
                        break
                    else:
                        self._log(f"  OCR cost slot={bar_index}: 方式={preprocess_name} 未识别到数字")
                except Exception as e:
                    self._log(f"  OCR cost slot={bar_index} 异常 ({preprocess_name}): {e}")
                    continue

        if self.debug:
            self._save_ocr_debug(
                cost_crop,
                f"cost_{raw.time_ms:08d}_slot_{bar_index}",
                value=result[0] if result else None,
                conf=result[1] if result else None,
            )

        return result[0] if result else None

    def _recognize_quantity(
        self,
        raw: RawAction,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int] = None,
    ) -> Optional[int]:
        """识别召唤物/道具右下角的数量数字。优先使用 ONNX 数字模型。"""
        if self.ocr is None:
            self._log(f"  OCR quantity slot={bar_index}: ocr 未初始化，跳过")
            return None
        bar_img = self._load_bar_image(raw)
        if bar_img is None:
            self._log(f"  OCR quantity slot={bar_index}: 未找到 DEPLOY_BAR 关键帧")
            return None
        # total_slots <= 12 时，active slot 的数量角标位置整体偏高 _ACTIVE_SELF_Y_SHIFT_PX 像素
        y_shift = -self._ACTIVE_SELF_Y_SHIFT_PX if (
            active_slot is not None
            and bar_index == active_slot
            and total_slots <= 12
        ) else 0
        qty_crop = self._crop_quantity_from_bar(
            bar_img, bar_index, total_slots, active_slot=active_slot, y_shift=y_shift
        )
        if qty_crop is None or qty_crop.size == 0:
            self._log(f"  OCR quantity slot={bar_index}: 数量 ROI 裁剪为空")
            return None

        result = None
        raw_text = ""
        proc = None
        try:
            proc = self._preprocess_quantity_strip(
                cv2.resize(qty_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                invert=True,
            )
        except Exception as e:
            self._log(f"  OCR quantity slot={bar_index} 预处理异常: {e}")

        # 1) 优先走 ONNX 数字模型
        if proc is not None:
            if self.debug:
                self._quantity_slot_debug_counter += 1
                self._save_ocr_debug(
                    proc,
                    f"qty_onnx_input_{raw.time_ms:08d}_slot_{bar_index}_{self._quantity_slot_debug_counter:04d}",
                )
            try:
                t0 = time.perf_counter()
                model_result = self._digit_recognizer.predict_quantity(proc)
                model_ms = (time.perf_counter() - t0) * 1000
                if model_result:
                    value, conf = model_result
                    if 0 <= value <= 30:
                        result = (value, conf)
                        self._log(
                            f"  OCR quantity slot={bar_index}: 模型={value} "
                            f"(conf={conf:.2f}, 耗时={model_ms:.2f}ms)"
                        )
            except Exception as e:
                self._log(f"  OCR quantity slot={bar_index} 模型异常: {e}")

        # 2) 模型失败则 fallback 到 OCR
        if result is None and proc is not None and self.ocr is not None:
            try:
                lines = self.ocr.recognize(proc, min_confidence=0.5)
                extracted = self._extract_quantity_from_ocr(lines, min_conf=0.5)
                if extracted:
                    result = (extracted[0], extracted[1])
                    raw_text = extracted[2]
                    self._log(
                        f"  OCR quantity slot={bar_index}: {result[0]} "
                        f"(conf={result[1]:.2f}, 原始='{raw_text}')"
                    )
            except Exception as e:
                self._log(f"  OCR quantity slot={bar_index} OCR 异常: {e}")

        if self.debug:
            self._quantity_slot_debug_counter += 1
            self._save_ocr_debug(
                qty_crop,
                f"qty_{raw.time_ms:08d}_slot_{bar_index}_{self._quantity_slot_debug_counter:04d}",
                value=result[0] if result else None,
                conf=result[1] if result else None,
            )

        return result[0] if result else None

    def _recognize_item_quantity(
        self,
        raw: RawAction,
        bar_index: int,
        total_slots: int,
        active_slot: Optional[int] = None,
    ) -> Optional[int]:
        """道具数量 ROI 与召唤物相同：头像右下角 40x40。"""
        return self._recognize_quantity(raw, bar_index, total_slots, active_slot=active_slot)

    def _read_quantity_crop(
        self, crop: np.ndarray
    ) -> Optional[Tuple[int, float, str]]:
        """对单个数量角标裁剪图先走 ONNX 模型，失败再 OCR fallback。

        返回 (quantity, confidence, source)，source 为 'model' 或 'ocr'。
        """
        if crop is None or crop.size == 0:
            return None

        # 1) ONNX 模型
        try:
            t0 = time.perf_counter()
            model_result = self._digit_recognizer.predict_quantity(crop)
            model_ms = (time.perf_counter() - t0) * 1000
            if model_result:
                value, conf = model_result
                if 0 <= value <= 30:
                    self._log(
                        f"  数量框模型读数: {value} "
                        f"(conf={conf:.2f}, 耗时={model_ms:.2f}ms)"
                    )
                    return value, conf, "model"
        except Exception as e:
            self._log(f"  数量框模型读数异常: {e}")

        # 2) OCR fallback
        if self.ocr is None:
            return None
        try:
            lines = self.ocr.recognize(crop, min_confidence=0.5)
            extracted = self._extract_quantity_from_ocr(lines, min_conf=0.5)
            if extracted:
                return extracted[0], extracted[1], "ocr"
        except Exception as e:
            self._log(f"  数量框 OCR fallback 异常: {e}")
        return None

    def _crop_cost_from_bar(
        self,
        bar_img: np.ndarray,
        bar_index: int,
        total_slots: int,
        y_shift: int = 0,
        active_slot: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """从整栏关键帧中裁剪指定 slot 的费用区域。

        当 total_slots >= 12 且存在数量 ROI 标定配置时，复用标定的水平中心位置，
        因为 deployment bar 形变时费用条和数量角标在水平方向同步移动。
        """
        if total_slots <= 0 or bar_index < 0 or bar_index >= total_slots:
            return None

        h_bar, w_bar = bar_img.shape[:2]
        window_width, window_height = self._recover_window_size_from_bar(bar_img)
        cell_w = window_width / 12 if total_slots <= 12 else window_width / total_slots
        bar_top = window_height * self._BAR_CAPTURE_TOP_RATIO

        # 费用条在窗口坐标 y=1390, h=36；bar_img 顶部的窗口 y = 1390 - 20（上移 20px）
        cost_y_window = window_height * constants.DEPLOY_BAR_COST_ROI_RATIOS[1]
        cost_h_window = window_height * constants.DEPLOY_BAR_COST_ROI_RATIOS[3]
        # 被拖拽的 active slot 整体向上偏移 _ACTIVE_SELF_Y_SHIFT_PX 像素，
        # 该偏移对 total_slots <= 12 和 > 12 均生效（>12 时 X 方向另有标定修正）。
        active_self_shift = (
            self._ACTIVE_SELF_Y_SHIFT_PX
            if active_slot is not None and active_slot == bar_index
            else 0
        )
        y1 = int(round(cost_y_window - bar_top + y_shift - active_self_shift))
        y2 = int(round(y1 + cost_h_window))

        # 优先使用 operator_cost logo 标定（total_slots > 12 时形变明显）
        cost_calibrated = None
        if active_slot is not None and total_slots > 12:
            cost_calibrated = self._get_calibrated_operator_cost_roi(
                total_slots, active_slot, bar_index
            )

        if cost_calibrated is not None:
            logo_x = cost_calibrated[0] * window_width
            rw = 53
            x1 = int(round(logo_x - 10 - rw / 2))
            x2 = x1 + rw
            self._log(
                f"  cost ROI 使用 operator_cost 标定: total={total_slots} "
                f"active={active_slot} target={bar_index} logo_x={logo_x:.1f}"
            )
        else:
            # 回退到数量 ROI 标定的水平中心（>=12 槽时形变明显）
            calibrated = None
            if active_slot is not None and total_slots >= 12:
                calibrated = self._get_calibrated_quantity_roi(
                    total_slots, active_slot, bar_index
                )

            if calibrated is not None:
                cx_ratio, _, half_w_ratio, _ = calibrated
                cx = cx_ratio * window_width
                half_w = half_w_ratio * window_width
                x1 = int(round(cx - half_w))
                x2 = int(round(cx + half_w))
                self._log(
                    f"  cost ROI 使用数量标定中心: total={total_slots} "
                    f"active={active_slot} target={bar_index} cx_ratio={cx_ratio:.4f}"
                )
            else:
                # 与脚本执行器 cost_recognition.recognize_operator_costs 保持一致：
                # 费用 ROI 从 slot 中心线开始向右截取 53px，而不是以中心线左右对称。
                cx = window_width - cell_w * (bar_index + 0.5)
                rw = 53
                x1 = int(round(cx))
                x2 = x1 + rw

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_bar, x2)
        y2 = min(h_bar, y2)

        if x2 <= x1 or y2 <= y1:
            return None
        return bar_img[y1:y2, x1:x2]

    def _save_ocr_debug(
        self,
        image: np.ndarray,
        prefix: str,
        value: Optional[int] = None,
        conf: Optional[float] = None,
    ):
        """保存 OCR 调试截图到会话目录，可选在图上标注识别结果与置信度。"""
        debug_dir = self._session_dir / "ocr_debug"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            canvas = image.copy()
            if value is not None or conf is not None:
                parts = []
                if value is not None:
                    parts.append(str(value))
                if conf is not None:
                    parts.append(f"({conf:.2f})")
                text = " ".join(parts) if parts else ""
                if text:
                    h, w = canvas.shape[:2]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                    pad = 4
                    cv2.rectangle(
                        canvas,
                        (0, 0),
                        (min(w, tw + pad * 2), min(h, th + pad * 2)),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        canvas,
                        text,
                        (pad, th + pad),
                        font,
                        scale,
                        (0, 0, 255),
                        thickness,
                        cv2.LINE_AA,
                    )
            canvas = self._bgra_to_bgr(canvas)
            path = debug_dir / f"{prefix}.png"
            ok, encoded = cv2.imencode(".png", canvas)
            if ok:
                path.write_bytes(encoded.tobytes())
                self._log(f"OCR 调试截图已保存: {path.name} ({canvas.shape[1]}x{canvas.shape[0]})")
            else:
                self._log(f"OCR 调试截图编码失败: {prefix}.png")
        except Exception as e:
            self._log(f"保存 OCR 调试截图失败: {e}")

    def _save_roi_debug(self, image: np.ndarray, prefix: str):
        """保存 ROI 可视化调试图到会话目录。"""
        debug_dir = self._session_dir / "roi_debug"
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            canvas = self._bgra_to_bgr(image)
            path = debug_dir / f"{prefix}.png"
            ok, encoded = cv2.imencode(".png", canvas)
            if ok:
                path.write_bytes(encoded.tobytes())
                self._log(f"ROI 调试图已保存: {path.name} ({image.shape[1]}x{image.shape[0]})")
            else:
                self._log(f"ROI 调试图编码失败: {prefix}.png")
        except Exception as e:
            self._log(f"保存 ROI 调试图失败: {e}")

    @staticmethod
    def _bgra_to_bgr(image: np.ndarray) -> np.ndarray:
        """若输入为 4 通道 BGRA，则转换为 3 通道 BGR；否则原样返回。"""
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    def _debug_visualize_bar_rois(
        self,
        bar_img: np.ndarray,
        total_slots: int,
        highlight_slot: Optional[int] = None,
        slot_labels: Optional[Dict[int, str]] = None,
        active_slot: Optional[int] = None,
    ) -> np.ndarray:
        """在整栏截图上画出每个 slot 的头像/费用/数量 ROI，并标注 OCR 结果。"""
        canvas = bar_img.copy()
        h_bar, w_bar = canvas.shape[:2]
        window_width, window_height = self._recover_window_size_from_bar(bar_img)
        cell_w = window_width / 12 if total_slots <= 12 else window_width / total_slots
        bar_top = window_height * self._BAR_CAPTURE_TOP_RATIO
        bar_center_y = window_height * self._BAR_CENTER_Y_RATIO
        cy_rel = bar_center_y - bar_top
        avatar_size = window_height * self._BAR_AVATAR_SIZE_RATIO
        y_offset = avatar_size * self._BAR_AVATAR_Y_OFFSET_RATIO

        cost_y_window = window_height * constants.DEPLOY_BAR_COST_ROI_RATIOS[1]
        cost_h_window = window_height * constants.DEPLOY_BAR_COST_ROI_RATIOS[3]

        slot_labels = slot_labels or {}

        for i in range(total_slots):
            cx = window_width - cell_w * (i + 0.5)
            cx_int = int(round(cx))
            is_highlight = i == highlight_slot
            thickness = 2 if is_highlight else 1

            # slot 中心线 + 索引
            cv2.line(canvas, (cx_int, 0), (cx_int, h_bar), (128, 128, 128), 1)
            cv2.putText(
                canvas,
                str(i),
                (cx_int - 5, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
                cv2.LINE_AA,
            )

            # operator_cost logo 标定（total_slots > 12 时修正头像/费用的 X）
            operator_cost_calibrated = None
            if active_slot is not None and total_slots > 12:
                operator_cost_calibrated = self._get_calibrated_operator_cost_roi(
                    total_slots, active_slot, i
                )

            # active slot 自身形变：被拖拽的 slot 整体向上移动固定像素
            # 注意：<=12 时头像偏移由 _compute_avatar_y_shift 负责，这里不再重复
            active_self_shift_px = 0
            cost_shift_px = 0
            qty_shift_px = 0
            if active_slot is not None and active_slot == i:
                if total_slots > 12:
                    active_self_shift_px = self._ACTIVE_SELF_Y_SHIFT_PX
                cost_shift_px = self._ACTIVE_SELF_Y_SHIFT_PX
                qty_shift_px = self._ACTIVE_SELF_Y_SHIFT_PX

            # 头像 ROI
            avatar_y_shift = self._compute_avatar_y_shift(
                bar_img, i, total_slots, active_slot
            )
            if operator_cost_calibrated is not None:
                avatar_cx = operator_cost_calibrated[0] * window_width - 40
            else:
                avatar_cx = cx
            crop_cx = int(round(avatar_cx))
            crop_cy = int(round(cy_rel + y_offset + avatar_y_shift - active_self_shift_px))
            crop_size = int(round(avatar_size))
            ax1 = max(0, crop_cx - crop_size // 2)
            ay1 = max(0, crop_cy - crop_size // 2)
            ax2 = min(w_bar, ax1 + crop_size)
            ay2 = min(h_bar, ay1 + crop_size)
            avatar_color = (0, 255, 255) if is_highlight else (0, 255, 0)
            cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), avatar_color, thickness)

            # 费用 ROI
            if operator_cost_calibrated is not None:
                logo_x = operator_cost_calibrated[0] * window_width
                ccx1 = int(round(logo_x - 10 - 53 / 2))
                ccx2 = ccx1 + 53
            else:
                # 回退到数量 ROI 标定的水平中心（>=12 槽时形变明显）
                cost_calibrated = None
                if active_slot is not None and total_slots >= 12:
                    cost_calibrated = self._get_calibrated_quantity_roi(
                        total_slots, active_slot, i
                    )
                if cost_calibrated is not None:
                    cx_ratio, _, half_w_ratio, _ = cost_calibrated
                    ccx = cx_ratio * window_width
                    c_half_w = half_w_ratio * window_width
                    ccx1 = int(round(ccx - c_half_w))
                    ccx2 = int(round(ccx + c_half_w))
                else:
                    ccx1 = int(round(cx))
                    ccx2 = min(w_bar, ccx1 + 53)
            cy1 = int(round(cost_y_window - bar_top - cost_shift_px))
            cy2 = int(round(cy1 + cost_h_window))
            cv2.rectangle(canvas, (ccx1, cy1), (ccx2, cy2), (0, 0, 255), thickness)

            # 数量 ROI：有标定时用标定框，否则用动态估算
            calibrated = None
            if active_slot is not None:
                calibrated = self._get_calibrated_quantity_roi(
                    total_slots, active_slot, i
                )
            if calibrated is not None:
                cx_ratio, cy_ratio, half_w_ratio, half_h_ratio = calibrated
                qx1 = int(round((cx_ratio - half_w_ratio) * window_width))
                qx2 = int(round((cx_ratio + half_w_ratio) * window_width))
                qy1 = int(round(
                    (cy_ratio - half_h_ratio) * window_height - bar_top
                ))
                qy2 = int(round(
                    (cy_ratio + half_h_ratio) * window_height - bar_top
                ))
                qx1 = max(0, qx1)
                qx2 = min(w_bar, qx2)
                qy1 = max(0, qy1)
                qy2 = min(h_bar, qy2)
            else:
                qx1 = int(round(cx))
                qx2 = min(w_bar, qx1 + int(round(cell_w / 2)))
                qy1 = int(round(window_height * self._QUANTITY_ROI_Y_RATIO - bar_top - qty_shift_px))
                qy2 = min(h_bar, qy1 + int(round(window_height * self._QUANTITY_ROI_H_RATIO)))
            cv2.rectangle(canvas, (qx1, qy1), (qx2, qy2), (255, 0, 0), thickness)

            # 在 slot 上方标注 OCR 结果/名称
            label = slot_labels.get(i)
            if label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.4
                thickness_label = 1
                lines = label.split("\n")
                max_w = 0
                total_h = 0
                for line in lines:
                    (tw, th), _ = cv2.getTextSize(line, font, scale, thickness_label)
                    max_w = max(max_w, tw)
                    total_h += th + 2
                lx1 = max(0, cx_int - max_w // 2)
                ly1 = max(total_h, cy1 - total_h - 2)
                cv2.rectangle(
                    canvas,
                    (lx1, ly1 - total_h),
                    (lx1 + max_w, ly1),
                    (0, 0, 0),
                    -1,
                )
                y_off = ly1 - total_h + 10
                for line in lines:
                    cv2.putText(
                        canvas,
                        line,
                        (lx1, y_off),
                        font,
                        scale,
                        (0, 255, 255),
                        thickness_label,
                        cv2.LINE_AA,
                    )
                    (tw, th), _ = cv2.getTextSize(line, font, scale, thickness_label)
                    y_off += th + 2

        if highlight_slot is not None:
            label = f"highlight slot[{highlight_slot}] total={total_slots}"
            if active_slot is not None:
                label += f" active={active_slot}"
            cv2.putText(
                canvas,
                label,
                (10, h_bar - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return canvas

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_bar_index(target_ref: str) -> Optional[int]:
        m = re.match(r"__slot_(\d+)__", target_ref)
        if m:
            return int(m.group(1))
        return None

    def _parse_click_ratio(self, target_ref: str) -> Optional[float]:
        """解析 target_ref 中的点击位置比例（相对窗口右边缘）。

        新录制使用 __click_<ratio>__；为兼容旧录制，__slot_N__ 会按 initial_total
        反推比例，使旧数据也能按当前实际总槽位数换算。
        """
        m = re.match(r"__click_([\d.]+)__", target_ref)
        if m:
            return float(m.group(1))
        bar_index = self._parse_bar_index(target_ref)
        if bar_index is not None:
            initial_total = self.raw.initial_operator_count + self.raw.initial_item_count
            if initial_total > 0:
                return (bar_index + 0.5) / initial_total
        return None
        for kid in raw.keyframe_ids:
            kf = self.raw.keyframes.get(kid)
            if kf is not None and kf.type == KeyframeType.DEPLOY_SLOT:
                return self._load_keyframe_image(kid)
        return None

    def _aligned_item_name(self, slot_index: int, prev_state: List[_SlotState]) -> str:
        """根据上一状态推断当前道具 slot 的名称，处理道具增加/减少带来的移位。

        道具 slot 始终位于最右侧（索引从右往左）。
          - 数量不变：直接沿用同一位置的道具名。
          - 数量减少：右侧道具耗尽消失，剩余道具向左移动，沿用 prev 中靠前
            的道具名。
          - 数量增加：新道具出现在最右侧，旧的道具整体向右移动一格。
        """
        prev_items = [s for s in prev_state if s.is_item]
        prev_count = len(prev_items)
        cur_count = self._remaining_item_count
        diff = cur_count - prev_count

        if diff > 0:
            # 有新道具出现在最右侧，前 diff 个 slot 使用新占位名
            if slot_index < diff:
                return self._item_name_for_bar_index(slot_index)
            prev_idx = slot_index - diff
        else:
            prev_idx = slot_index

        if 0 <= prev_idx < prev_count:
            name = prev_items[prev_idx].name
            if name:
                return name
        return self._item_name_for_bar_index(slot_index)

    def _item_name_for_bar_index(self, bar_index: int) -> str:
        return f"__item_{bar_index}__"

    def _ensure_operator(self, name: str):
        if name not in self._operators:
            self._operators.append(name)

    def _ensure_item(self, name: str):
        if not any(it.name == name for it in self._items):
            self._items.append(ItemInfo(name=name, charges=1))

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def _build_script(self) -> ScriptModel:
        # 补充用户填写的初始道具（即使未被使用），并按 OCR 数量/使用次数设置 charges。
        # 如果某个 bar_index 已通过名称卡 OCR 得到真实名称，优先使用真实名称。
        # 无限道具同时视为初始道具（charges=1）和召唤物（用于 ADD_SUMMON 生命周期）。
        infinite_original_indices = set(self._infinite_items.keys())
        bar_index_to_name = {idx: nm for nm, idx in self._item_bar_index.items()}
        for bar_index in range(self._item_count_hint):
            is_infinite = bar_index in infinite_original_indices
            if is_infinite:
                # 无限道具与 summons 列表共用 _infinite_items 中的名称
                name = self._infinite_items[bar_index].name
            else:
                name = bar_index_to_name.get(bar_index) or self._item_name_for_bar_index(bar_index)
            self._ensure_item(name)
            self._item_bar_index[name] = bar_index
            if is_infinite:
                # 无限道具在 items 中只记录初始 1 个，后续通过 ADD_SUMMON 补充
                for it in self._items:
                    if it.name == name:
                        it.charges = 1
                        break
            else:
                usage = self._item_usage_counts.get(name, 0)
                qty = self._item_initial_quantity.get(name)
                for it in self._items:
                    if it.name == name:
                        it.charges = max(1, qty if qty is not None else usage)
                        break

        # 道具按实际 bar_index 从右到左排序，确保 OperatorPool 布局与录制时一致
        self._items.sort(
            key=lambda it: self._item_bar_index.get(it.name, 0),
            reverse=True,
        )

        # 构建 summons 列表
        infinite_names = {it.name for it in self._infinite_items.values()}
        for name in sorted(self._summon_deploy_counts.keys(), key=lambda n: self._summon_deploy_counts.get(n, 0)):
            cost = self._summon_costs.get(name, 0)
            # 无限道具已在 items 中作为初始道具，summons 中只注册不重复初始放置
            initial_charges = 0
            self._summons.append(SummonInfo(name=name, cost=cost, initial_charges=initial_charges))

        # 干员直接按初始部署区头像解析出的视觉顺序回填（从左到右），
        # 助战干员在初始布局中本就位于干员区域最右侧，因此自然在列表末尾。
        if self._initial_operator_order:
            seen = set()
            ordered: List[str] = []
            for name in self._initial_operator_order:
                if name in self._operators and name not in seen:
                    ordered.append(name)
                    seen.add(name)
            for name in self._operators:
                if name not in seen:
                    ordered.append(name)
            self._operators = ordered
        else:
            # 无初始顺序时回退：按费用排序，助战位于末尾
            support_names = list(self._support_operator_names())
            normal_names = [n for n in self._operators if n not in support_names]

            def _operator_sort_key(name: str):
                info = self._operator_info.get(name)
                cost = info.cost if info is not None and info.cost is not None else 0
                visual_index = self._operator_index.get(name, -1)
                return (cost, -visual_index, name)

            normal_names.sort(key=_operator_sort_key)
            self._operators = normal_names + support_names

        # 合并用户提示中的召唤物绑定（若存在）
        bindings: List[SummonBinding] = []
        raw_bindings = self.raw.hints.get("summon_bindings")
        if isinstance(raw_bindings, list):
            for b in raw_bindings:
                if isinstance(b, dict):
                    bindings.append(SummonBinding(**b))
                elif isinstance(b, SummonBinding):
                    bindings.append(b.model_copy())

        script = ScriptModel(
            stage_code=self.raw.stage_code,
            grid_rows=self.raw.grid_rows,
            grid_cols=self.raw.grid_cols,
            operators=self._operators,
            items=self._items,
            summons=self._summons,
            summon_bindings=bindings,
            actions=self._actions,
        )
        script.sort_actions()
        return script
