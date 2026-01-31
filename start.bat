@echo off
title Background-Dodger Launcher
cls
echo ===================================================
echo   Background-Dodger AI Launcher
echo ===================================================
echo.
echo Please select what you want to run:
echo.
echo   [1] Full Experience (Image + Video Gen)  WARNING: Uses ~25GB RAM/Swap
echo   [2] Image Generator Only (Preferred)
echo   [3] Video Generator Only (Preferred)
echo   [4] Frontend Only (Background Remover)
echo.
set /p option="Enter option (1-4): "

:: Start Frontend (Using Python since npx is broken on your system)
:: This runs a simple HTTP server on port 8080
start "Frontend" cmd /k "python -m http.server 8080"
echo Started Frontend on port 8080...

if "%option%"=="1" goto START_ALL
if "%option%"=="2" goto START_FLUX
if "%option%"=="3" goto START_VIDEO
if "%option%"=="4" goto END

:START_ALL
echo Starting BOTH servers...
start "FLUX Server" cmd /k "cd server && python flux_server.py"
start "Video Server" cmd /k "cd server && python video_server.py"
goto INFO

:START_FLUX
echo Starting Image Generator server...
start "FLUX Server" cmd /k "cd server && python flux_server.py"
goto INFO

:START_VIDEO
echo Starting Video Generator server...
start "Video Server" cmd /k "cd server && python video_server.py"
goto INFO

:INFO
echo.
echo ===================================================
echo   System Running!
echo   Open: http://localhost:8080
echo ===================================================
echo.
echo Close this window to exit.
pause >nul
goto END

:END
exit
