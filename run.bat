@echo off
title MindGuard AI - Early Distress Detection Assistant

echo ============================================================
echo   MindGuard AI - Early Distress Detection Assistant
echo   Powered by IBM watsonx.ai - Granite Models - RAG
echo ============================================================
echo.

set PYTHONUTF8=1
set WATSONX_API_KEY=2f_HxjEGmMQopogbgpOEUtSBprds9kC45n35-Bb06Vna
set WATSONX_PROJECT_ID=12f032c6-400d-4253-b52a-40dcacf7a7d6
set WATSONX_URL=https://us-south.ml.cloud.ibm.com
set GRANITE_MODEL_ID=ibm/granite-13b-instruct-v2

echo [1/2] Installing dependencies...
"C:\Users\KSSEM\AppData\Local\Python\bin\python.exe" -m pip install flask ibm-watsonx-ai --quiet

echo [2/2] Starting MindGuard AI...
echo.
echo  Open your browser and go to:
echo  http://localhost:5000
echo.
echo  Press Ctrl+C to stop the server.
echo ============================================================

"C:\Users\KSSEM\AppData\Local\Python\bin\python.exe" app.py

pause
