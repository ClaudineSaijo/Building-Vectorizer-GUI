@echo off
chcp 65001 >nul
cd /d "D:\All4DS\BISHE"
echo 启动每日训练/测试队列（06:30-13:50 / 16:00-23:00 双窗口，窗口间自动休眠）
echo 日志：runs\dayplan\day.log
start "BISHE day runner" /min "D:\All4DS\BISHE\.venv\Scripts\python.exe" "scripts\day_runner.py"
timeout /t 2 >nul
