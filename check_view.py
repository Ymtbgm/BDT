import json
from collections import defaultdict

from pathlib import Path

with open(Path(__file__).parent / "core" / "resource" / "levels.json", "r", encoding="utf-8") as f:
    levels = json.load(f)

size_to_views = defaultdict(list)

for lv in levels:
    view = lv.get("view", [[], []])
    v0 = tuple(view[0]) if len(view) > 0 else None
    size = (lv.get("width"), lv.get("height"))
    if v0:
        size_to_views[size].append(v0)

# 统计每个尺寸最常见的 view
for size, views in sorted(size_to_views.items()):
    from collections import Counter
    counter = Counter(views)
    most_common = counter.most_common(1)[0]
    print(f"{size[0]}x{size[1]}: {len(views)} 个关卡, 最常见 view={list(most_common[0])} ({most_common[1]} 个)")
