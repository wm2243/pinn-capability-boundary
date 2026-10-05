@echo off
REM D1 rational 族 beta 扫描全量（44 runs，GPU，约 5 小时）
REM 断点续跑：每 5000 步自动存档；中断后直接重新双击本文件即可继续，已完成 run 自动跳过。
cd /d "%~dp0"
set D1_WALLCAP=0
"D:\workenvironmentAnaconda\envs\pinns\python.exe" exp_D1_rational_beta_scan.py
echo.
echo ==== 全部结束（若中途被中断，重新运行本 bat 会继续）====
pause
