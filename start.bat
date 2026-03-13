@echo off
echo Starting AI Face Analytics System...

start cmd /k "echo Starting FastAPI Backend... && cd /d %~dp0 && uvicorn app.main:app --host 127.0.0.1 --port 8001"

start cmd /k "echo Starting React Frontend... && cd /d %~dp0\frontend && npm run dev"

echo Both servers are starting in separate windows.
echo - Backend API: http://localhost:8001/docs
echo - React UI:    http://localhost:5173
echo.
pause

