from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit


class GuideTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        guide_text = QTextEdit()
        guide_text.setReadOnly(True)
        guide_text.setHtml(
            """
            <h2>Arknights Auto 使用指南</h2>
            <h3>1. 脚本编辑</h3>
            <p>在「脚本编辑」标签页中创建或修改 JSON 脚本。</p>
            <ul>
                <li><b>关卡名</b>：用于 OCR 校验，确保进入了正确的关卡。</li>
                <li><b>关卡代号</b>：用于查询相机位置（levels.json），大多时候与关卡名一致，但危机合约类关卡往往有特殊命名，目前以level_crisis开头。</li>
                <li><b>地图行列</b>：地图的总行数和总列数，左上为(0,0)，向下加一行，向右加一列，可视化可以查阅各种MAP网站。</li>
                <li><b>干员列表</b>：按<span style="color: red;">进入游戏后初始部署栏顺序从左往右</span>逐个添加。</li>
                <li><b>道具列表</b>：同上添加道具名和可用次数。</li>
            </ul>
            <h3>2. 时间轴操作</h3>
            <p>操作类型包括：deploy（部署）、retreat（撤退）、skill（技能）、
            speed_up（加速）、speed_down（减速）、pause（暂停）、add_item（部署区道具）。</p>
            <p><b>格子格式</b>：行,列（例如 3,2）。</p>
            <p><b>方向</b>：up / down / left / right。</p>
            <p><b>装置</b>：勾选 is_object 表示目标为场上装置/衍生物，不通过部署栏选中。</p>
            <p><b>add_item（部署区新增道具）</b>：当击杀敌人，获得召唤物等获得额外可部署道具时使用。
            格子填写"序号,次数"，其中序号为该道具在道具区域中的从左到右位置（0 表示紧挨着干员的最左侧，数字越大越靠右），次数为可使用次数。</p>
            <p>为保证操作精度，所有操作在子弹时间下进行，<span style="color: red;">请保证部署区始终存在单位，即哪怕单人图，也请携带1名不下场干员</span></p>

            <h3>3. 脚本执行</h3>
            <p>在「脚本执行」标签页中选择脚本并运行，运行时不要遮挡游戏屏幕。</p>
            <ul>
                <li><b>无限凸图</b>：脚本结束后自动重新挑战。</li>
                <li><b>漏怪检测</b>：检测到漏怪后自动退出并补打一次（仅一次，不会无限循环）。</li>
                <li><b>Debug</b>：输出调试日志，仅调试BUG使用，平常开启会降低性能。</li>
                <li><b>直接开始作战</b>：跳过 OCR 查找关卡和"开始行动"点击。适用于已手动进入干员编队界面的场景，此时助战参数不可用，请自行选好助战。</li>
                <li><b>突袭模式</b>：进关卡前会选择突袭。</li>
                <li><b>助战参数</b>：不借用则不勾选，好友位从左到右依次为0到9，点击助战后不做任何移动，技能从左到右依此为1到3，模组从左到右依此为1到3。</li>
                <li><b>键位设置</b>：<span style="color: red;">键位务必和游戏中对应快捷键位一致</span>，可以下拉选择，也可以输入。运行一次脚本后自动保存。</li>
            </ul>
            <h3>4. 资源更新</h3>
            <p>在「资源更新」标签页中可以上传新的 levels.json 文件。</p>
            levels.json为游戏解包关卡资源，包含了不同关卡相机位置，为了精准对齐格子需要更新加载。<br>
            资源可在https://github.com/yuanyan3060/ArknightsGameResource中获取，请为该解包和格子对齐项目点上star吧！<br>
            <h3>5. 快捷键</h3>
            <ul>
                <li><b>F11</b>：暂停/恢复脚本</li>
                <li><b>F12</b>：紧急暂停脚本并暂停游戏</li>
            </ul>
            <h3>6. 其他注意事项</h3>
            <ul>
                <li><b>UI设置</b>：UI设置请采用默认的90大小，否则脚本无法有效执行。</li>
                <li><b>管理员权限</b>：请以管理员模式启动，否则游戏无法接受键位和鼠标操作。</li>
            <ul>
            """
        )
        layout.addWidget(guide_text)
