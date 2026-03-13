@echo off
echo Starting AI Face Analytics System...

start cmd /k "echo Starting FastAPI Backend... && cd /d %~dp0 && uvicorn app.main:app --host 0.0.0.0 --port 8000"

start cmd /k "echo Starting React Frontend... && cd /d %~dp0\frontend && npm run dev"

echo Both servers are starting in separate windows.
echo - Backend API: http://localhost:8000/docs
echo - React UI:    http://localhost:5173
echo.
pause

