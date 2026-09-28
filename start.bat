@echo off
chcp 65001 >nul
cd /d "%~dp0backend"
echo Starting doc-knowledge-base at http://127.0.0.1:8765 ...
"C:\Users\Niko\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8765
