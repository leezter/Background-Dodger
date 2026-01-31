@echo off
echo Starting Background-Dodger servers...
echo.

:: Start frontend server
echo [1/3] Starting frontend server (port 8080)...
start "Frontend" cmd /k "npx http-server . -p 8080"

:: Start FLUX image server
echo [2/3] Starting FLUX image server (port 8000)...
start "FLUX Server" cmd /k "cd server && python flux_server.py"

:: Start CogVideoX video server
echo [3/3] Starting CogVideoX video server (port 8001)...
start "Video Server" cmd /k "cd server && python video_server.py"

echo.
echo All servers starting in separate windows!
echo.
echo   Frontend:    http://localhost:8080
echo   FLUX API:    http://localhost:8000
echo   Video API:   http://localhost:8001
echo.
echo Close this window or press any key to exit.
pause >nul
