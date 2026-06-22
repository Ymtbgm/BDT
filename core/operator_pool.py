from typing import List, Optional, Tuple, Dict

from models.script_schema import ItemInfo


class OperatorPool:
    """管理干员与道具在部署栏的动态位置。

    道具优先排列在部署栏最右侧（索引 0, 1, ...），干员排在道具之后。
    道具按次数使用，剩余次数 > 0 时留在部署区；次数归零后从栏位移除。
    """

    def __init__(self, window_width: int, window_height: int,
                 operators: List[str], items: Optional[List[ItemInfo]] = None):
        self.window_width = window_width
        self.window_height = window_height
        self.operators = list(operators)
        self.items = list(items) if items else []
        # 道具当前剩余次数
        self._item_charges: Dict[str, int] = {it.name: it.charges for it in self.items}
        # 可用道具名称列表：与干员列表统一为"从左到右"的视觉顺序。
        # _calc_bar_positions 中 i=0 对应最右侧，因此内部列表需要反转，
        # 使得 items[0] 对应道具区域最左侧（紧挨着干员）。
        self._available_items: List[str] = [it.name for it in reversed(self.items)]
        # 部署区中保存的是干员在初始列表中的索引（初始序号），始终升序排列
        self._bar_indices: List[int] = list(range(len(operators)))
        self._deployed: Dict[str, Tuple[int, int]] = {}
        self._left = 0
        self._top = 0
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
        """
        total = len(self._bar_indices) + len(self._available_items)
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

    def _name_to_index(self, operator_name: str) -> int:
        return self.operators.index(operator_name)

    def get_deploy_pos(self, name: str) -> Optional[Tuple[int, int]]:
        """获取干员或道具在部署栏中的像素坐标，若不在部署区则返回 None。"""
        if self._is_item(name):
            if name not in self._available_items:
                return None
            pos_in_bar = self._available_items.index(name)
            return self._bar_positions.get(pos_in_bar)
        # 干员：operators 列表顺序从左到右对应部署栏
        idx = self._name_to_index(name)
        if idx not in self._bar_indices:
            return None
        total = len(self._bar_indices) + len(self._available_items)
        pos_in_bar = (total - 1) - self._bar_indices.index(idx)
        return self._bar_positions.get(pos_in_bar)

    def get_deployed_grid(self, name: str) -> Optional[Tuple[int, int]]:
        """获取干员当前部署的地图格子 (row, col)。"""
        return self._deployed.get(name)

    def deploy(self, name: str, grid: Tuple[int, int]):
        """执行部署逻辑。

        - 道具：扣除 1 次使用次数；次数归零时从部署区移除。
        - 干员：从部署区移除，记录实际部署位置。
        """
        if self._is_item(name):
            if name not in self._available_items:
                return
            self._item_charges[name] -= 1
            if self._item_charges[name] <= 0:
                self._available_items.remove(name)
                self._recalc()
            # 道具部署后不记录位置（仍在部署区或已移除）
            return

        idx = self._name_to_index(name)
        if idx in self._bar_indices:
            self._bar_indices.remove(idx)
            self._recalc()
        self._deployed[name] = grid

    def retreat(self, name: str):
        """干员撤退：从已部署移除，按初始序号插入回部署区。
        道具不能撤退。"""
        if self._is_item(name):
            return
        if name in self._deployed:
            del self._deployed[name]
        idx = self._name_to_index(name)
        if idx not in self._bar_indices:
            self._bar_indices.append(idx)
            self._bar_indices.sort()
            self._recalc()

    def is_available(self, name: str) -> bool:
        if self._is_item(name):
            return name in self._available_items
        idx = self._name_to_index(name)
        return idx in self._bar_indices

    def is_deployed(self, name: str) -> bool:
        return name in self._deployed

    def get_bar_index_pos(self, index: int) -> Optional[Tuple[int, int]]:
        """获取部署栏指定索引（0 为最右侧）的像素坐标。"""
        return self._bar_positions.get(index)

    def add_extra_item(self, name: str, bar_index: int, charges: int):
        """在部署区动态新增一个额外道具（如关卡中击杀敌人获得的装置）。

        Args:
            name: 道具名称
            bar_index: 在道具区域中的从左到右序号（0=道具最左侧，即紧挨着干员右侧；数字越大越靠右）
            charges: 可使用次数
        """
        # 用户习惯从左到右编号，但内部 _available_items 是从右到左排列
        # 因此需要将用户的从左到右序号转换为内部插入位置
        insert_pos = len(self._available_items) - bar_index
        if name in self._item_charges:
            # 已存在则更新次数并调整位置
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
            "deployed": dict(self._deployed),
        }
