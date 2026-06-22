import sys

# 显式导入 gui 包，确保 PyInstaller 能静态分析到 GUI 依赖
# （entry.py 中的条件导入在 else 分支，PyInstaller 可能追踪不到）
import gui  # noqa: F401


def main():
    args = sys.argv[1:]
    if "--run-script" in args:
        # 后端模式：重组参数后调用 main.py 的逻辑
        idx = args.index("--run-script")
        # 把 --run-script 后面的参数直接作为 sys.argv 传给后端
        new_argv = [sys.argv[0]] + args[idx + 1:]
        sys.argv = new_argv
        import asyncio
        from main import main as backend_main
        asyncio.run(backend_main())
    else:
        # GUI 模式
        from gui.app import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
