import os
import sys

# 在导入任何可能使用 OpenMP/MKL 的库（torch/onnxruntime/paddle）之前设置环境变量，
# 避免 PyInstaller 打包后因多份 OpenMP 运行时冲突导致推理回退到单线程。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_BLOCKTIME", "0")
try:
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(os.cpu_count() or 4, 8))))
except Exception:
    pass

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
