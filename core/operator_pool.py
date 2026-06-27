from typing import List, Optional, Tuple, Dict

from models.script_schema import ItemInfo, SummonInfo


class OperatorPool:
    """管理干员、召唤物与道具在部署栏的动态位置。

    道具优先排列在部署栏最右侧（索引 0, 1, ...）。
    干员与召唤物按费用从低到高从左到右排列；未识别费用时，干员回退到初始序号排序。
    助战干员已包含在 operators 列表中（通常为最后一名），不参与费用排序，固定排在干员区域最右侧。
    道具按次数使用，剩余次数 > 0 时留在部署区；次数归零后从栏位移除。
    """

    def __init__(
        self,
        window_width: int,
        window_height: int,
        operators: List[str],
        items: Optional[List[ItemInfo]] = None,
        summons: Optional[List[SummonInfo]] = None,
        support_count: int = 0,
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.operators = list(operators)
        self.items = list(items) if items else []
        self.support_count = max(0, support_count)

        # 道具当前剩余次数
        self._item_charges: Dict[str, int] = {it.name: it.charges for it in self.items}
        # 可用道具名称列表：与干员列表统一为"从左到右"的视觉顺序。
        # _calc_bar_positions 中 i=0 对应最右侧，因此内部列表需要反转，
        # 使得 items[0] 对应道具区域最左侧（紧挨着干员）。
        self._available_items: List[str] = [it.name for it in reversed(self.items)]

        # 干员在初始列表中的索引；未设置费用时按索引升序，设置费用后按费用排序
        self._bar_indices: List[int] = list(range(len(operators)))
        # 干员名称 -> 费用；未识别费用时 get_deploy_pos 会回退到初始序号
        self._operator_costs: Dict[str, int] = {}

        self._summons: Dict[str, int] = {}
        # 特殊召唤物剩余数量：{name: charges}，ADD_SUMMON 可以增加多个，
        # 部署一次消耗 1，数量归零时才从部署栏移除。
        self._summon_charges: Dict[str, int] = {}
        self._deployed: Dict[str, Tuple[int, int]] = {}
        self._left = 0
        self._top = 0
        self.register_summons(summons or [])

        self._bar_positions: Dict[int, Tuple[int, int]] = {}
        self._recalc()

    def set_window_offset(self, left: int, top: int):
        self._left = left
        self._top = top
        self._recalc()

    def update_window_size(self, width: int, height: int):
        self.window_width = width
        self.window_height = height
        self._recalc()

    def _recalc(self):
        """重新计算部署栏坐标。"""
        self._bar_positions = self._calc_bar_positions()

    def _calc_bar_positions(self) -> Dict[int, Tuple[int, int]]:
        """根据当前部署区人数 + 道具数计算坐标：
        - 纵坐标固定比例 1480/1600
        - <=12 时头像宽度 = 窗口宽 / 12
        - >12 时头像宽度 = 窗口宽 / 总人数
        - 从右朝左排列，索引 0 对应最右侧（道具优先）
        - 助战干员已包含在 operators 中，固定排在最右侧且不参与费用排序
        """
        total = len(self._bar_indices) + len(self._available_items) + sum(
            1 for c in self._summon_charges.values() if c > 0
        )
        if total == 0:
            return {}
        w, h = self.window_width, self.window_height
        bar_y = int(h * 1480 / 1600)
        cell_w = w / 12 if total <= 12 else w / total
        positions = {}
        for i in range(total):
            cx = w - cell_w * (i + 0.5)
            positions[i] = (self._left + int(cx), self._top + bar_y)
        return positions

    def _is_item(self, name: str) -> bool:
        return name in self._item_charges

    def _is_summon(self, name: str) -> bool:
        return name in self._summons

    def is_summon(self, name: str) -> bool:
        return self._is_summon(name)

    def _summon_available(self, name: str) -> bool:
        return self._is_summon(name) and self._summon_charges.get(name, 0) > 0

    def _name_to_index(self, operator_name: str) -> int:
        return self.operators.index(operator_name)

    def _is_support_operator(self, name: str) -> bool:
        """判断干员是否为 operators 列表末尾的助战干员。"""
        if self.support_count <= 0:
            return False
        idx = self._name_to_index(name)
        return idx >= len(self.operators) - self.support_count

    def _get_unit_cost(self, name: str) -> int:
        """返回单位用于排序的费用：召唤物用脚本费用，干员优先用 OCR 费用，未识别则回退到初始序号。"""
        if self.is_summon(name):
            return self._summons[name]
        idx = self._name_to_index(name)
        return self._operator_costs.get(name, idx)

    def _get_unit_sort_key(self, name: str):
        """返回单位在部署栏左侧排序时的完整键：费用升序，同费用干员按初始序号，召唤物排在干员右侧，助战固定在最右。"""
        if self.is_summon(name):
            return (self._summons[name], len(self.operators), 1, name)
        idx = self._name_to_index(name)
        if self._is_support_operator(name):
            # 助战不参与费用排序，固定排在最右侧
            return (float("inf"), idx, 0, "")
        return (self._get_unit_cost(name), idx, 0, "")

    def _get_left_units(self) -> List[str]:
        """返回左侧干员+可用召唤物按费用从左到右排序后的名称列表；助战干员固定排在最右侧。"""
        units = []
        support_names = []
        for idx in self._bar_indices:
            name = self.operators[idx]
            if self._is_support_operator(name):
                support_names.append(name)
            else:
                units.append((self._get_unit_sort_key(name), name))
        for name, charges in self._summon_charges.items():
            if charges > 0:
                units.append((self._get_unit_sort_key(name), name))
        units.sort(key=lambda x: x[0])
        # 助战固定在最右侧（紧挨道具左侧）
        return [name for _, name in units] + support_names

    def get_deploy_pos(self, name: str) -> Optional[Tuple[int, int]]:
        """获取干员、召唤物或道具在部署栏中的像素坐标，若不在部署区则返回 None。"""
        if self._is_item(name):
            if name not in self._available_items:
                return None
            pos_in_bar = self._available_items.index(name)
            return self._bar_positions.get(pos_in_bar)

        if self.is_summon(name):
            if not self._summon_available(name):
                return None
            left_units = self._get_left_units()
            pos_in_bar = (len(left_units) + len(self._available_items) - 1) - left_units.index(name)
            return self._bar_positions.get(pos_in_bar)

        # 干员
        idx = self._name_to_index(name)
        if idx not in self._bar_indices:
            return None
        left_units = self._get_left_units()
        pos_in_bar = (len(left_units) + len(self._available_items) - 1) - left_units.index(name)
        return self._bar_positions.get(pos_in_bar)

    def get_deployed_grid(self, name: str) -> Optional[Tuple[int, int]]:
        """获取干员当前部署的地图格子 (row, col)。"""
        return self._deployed.get(name)

    def deploy(self, name: str, grid: Tuple[int, int]):
        """执行部署逻辑。

        - 道具：扣除 1 次使用次数；次数归零时从部署区移除。
        - 干员/召唤物：从部署区移除，记录实际部署位置。
        """
        if self._is_item(name):
            if name not in self._available_items:
                return
            self._item_charges[name] -= 1
            if self._item_charges[name] <= 0:
                self._available_items.remove(name)
                self._recalc()
            return

        if self.is_summon(name):
            if not self._summon_available(name):
                return
            self._summon_charges[name] -= 1
            if self._summon_charges[name] <= 0:
                del self._summon_charges[name]
            self._recalc()
            self._deployed[name] = grid
            return

        idx = self._name_to_index(name)
        if idx in self._bar_indices:
            self._bar_indices.remove(idx)
            self._recalc()
        self._deployed[name] = grid

    def retreat(self, name: str):
        """干员撤退：从已部署移除，按费用插入回部署区。
        召唤物撤退后不再回到部署栏（由用户通过 ADD_SUMMON 再次获得）。
        道具不能撤退。"""
        if self._is_item(name):
            return
        if name in self._deployed:
            del self._deployed[name]

        if self.is_summon(name):
            # 召唤物撤退后不自动回到部署栏，需用户通过 ADD_SUMMON 再次获得。
            self._recalc()
            return

        idx = self._name_to_index(name)
        if idx not in self._bar_indices:
            self._bar_indices.append(idx)
            self._sort_bar_indices()
            self._recalc()

    def is_available(self, name: str) -> bool:
        if self._is_item(name):
            return name in self._available_items
        if self.is_summon(name):
            return self._summon_available(name)
        idx = self._name_to_index(name)
        return idx in self._bar_indices

    def is_deployed(self, name: str) -> bool:
        return name in self._deployed

    def get_bar_index_pos(self, index: int) -> Optional[Tuple[int, int]]:
        """获取部署栏指定索引（0 为最右侧）的像素坐标。"""
        return self._bar_positions.get(index)

    def register_summons(self, summons: List[SummonInfo]):
        """预注册脚本中定义的所有召唤物（此时还不在部署栏中）。"""
        self._summons = {s.name: s.cost for s in summons}
        self._summon_charges.clear()
        self._recalc()

    def activate_summon(self, name: str, charges: int = 1):
        """执行 ADD_SUMMON 后调用，让指定召唤物真正进入部署栏。"""
        if name not in self._summons:
            raise ValueError(f"未注册的召唤物: {name}")
        if charges <= 0:
            return
        self._summon_charges[name] = self._summon_charges.get(name, 0) + charges
        self._recalc()

    def set_operator_costs(self, costs: Dict[str, int]):
        """注入 OCR 识别到的干员费用，并按费用重排部署栏。"""
        self._operator_costs = dict(costs)
        self._sort_bar_indices()
        self._recalc()

    def set_support_count(self, support_count: int):
        """运行时动态设置助战干员数量（用于直接开始作战时手动借用的检测）。"""
        self.support_count = max(0, support_count)
        self._sort_bar_indices()
        self._recalc()

    def _sort_bar_indices(self):
        """按费用对 _bar_indices 排序（助战干员保持在末尾，不参与排序）。"""
        support_threshold = len(self.operators) - self.support_count
        normal_indices = [idx for idx in self._bar_indices if idx < support_threshold]
        support_indices = [idx for idx in self._bar_indices if idx >= support_threshold]
        normal_indices.sort(key=lambda idx: (self._get_unit_cost(self.operators[idx]), idx))
        self._bar_indices = normal_indices + support_indices

    def add_extra_item(self, name: str, bar_index: int, charges: int):
        """在部署区动态新增一个额外道具（如关卡中击杀敌人获得的装置）。

        Args:
            name: 道具名称
            bar_index: 在道具区域中的从左到右序号（0=道具最左侧，即紧挨着干员右侧；数字越大越靠右）
            charges: 可使用次数
        """
        insert_pos = len(self._available_items) - bar_index
        if name in self._item_charges:
            self._item_charges[name] = charges
            if name in self._available_items:
                self._available_items.remove(name)
            self._available_items.insert(insert_pos, name)
        else:
            self._item_charges[name] = charges
            self._available_items.insert(insert_pos, name)
        self._recalc()

    def state_summary(self) -> dict:
        return {
            "items": {n: c for n, c in self._item_charges.items() if c > 0},
            "bar_operators": [self.operators[i] for i in self._bar_indices],
            "bar_summons": {n: c for n, c in self._summon_charges.items() if c > 0},
            "deployed": dict(self._deployed),
        }
