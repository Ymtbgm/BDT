@echo off
chcp 65001 >nul
echo ============================================
echo  正在打包 Arknights Auto...
echo ============================================

REM 清理旧构建
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

REM 使用 PyInstaller 打包
REM --onedir: 单目录模式，启动快，方便用户替换资源
REM --windowed: 隐藏控制台窗口
REM --exclude-module: 排除不需要的重量级依赖以缩减体积
pyinstaller --clean ^
    --onedir ^
    --windowed ^
    --name "ArknightsAuto" ^
    --distpath "dist" ^
    --workpath "build" ^
    --hidden-import paddleocr ^
    --hidden-import paddlex ^
    --hidden-import paddle ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import pydantic ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module scipy ^
    --exclude-module pandas ^
    --exclude-module matplotlib ^
    --exclude-module skimage ^
    --exclude-module sklearn ^
    --collect-all PyQt6 ^
    --collect-all PyQt6.QtCore ^
    --collect-all PyQt6.QtGui ^
    --collect-all PyQt6.QtWidgets ^
    --collect-all cv2 ^
    --collect-all numpy ^
    --collect-all mss ^
    entry.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 打包失败，请检查上面的错误信息。
    pause
    exit /b 1
)

REM 将用户可替换的资源/脚本放到 exe 同级目录，方便后续更新
echo.
echo 正在复制资源文件到输出目录...
xcopy "core\resource" "dist\ArknightsAuto\core\resource\" /E /I /Y >nul
xcopy "scripts" "dist\ArknightsAuto\scripts\" /E /I /Y >nul
copy "example_script.json" "dist\ArknightsAuto\" /Y >nul

echo.
echo ============================================
echo  打包完成！
echo  输出目录: dist\ArknightsAuto\
echo ============================================
pause
