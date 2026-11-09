@echo off
chcp 65001 > nul
echo ========================================================
echo    Starting Thai Coin Counter & Detector Server
echo ========================================================
.\.venv\Scripts\python.exe run.py
pause
