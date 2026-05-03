@echo off
REM Local regression runner for Windows.
REM Run this BEFORE every commit to scripts/ or turing/.
REM
REM Usage:
REM   tests\run_local.bat               # tier=structural (no deps)
REM   tests\run_local.bat active        # also tries transformers class imports
REM   tests\run_local.bat full          # also tries config-only network loads

setlocal
set TIER=%1
if "%TIER%"=="" set TIER=structural

echo ===============================================================
echo Running local regression suite (tier=%TIER%)
echo ===============================================================
python tests\regression.py --tier %TIER%
set RC=%ERRORLEVEL%

if %RC% NEQ 0 (
    echo.
    echo *** REGRESSION FAILED -- DO NOT COMMIT ***
    exit /b %RC%
) else (
    echo.
    echo OK -- safe to commit.
    exit /b 0
)
