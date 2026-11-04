@echo off
REM SMC bot on Windows with data from the running MetaTrader 5 terminal.
REM Usage: run_windows.bat [backtest|experiments|analyze ...|news|sessions|presets|config]   (default: backtest)
REM First run creates .venv and installs dependencies. Needs Python 3.11-3.13 (python.org).
cd /d "%~dp0"
if not exist .venv (
    py -3 -m venv .venv || python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements-windows.txt
)
if not exist .env copy .env.example .env
REM load KEY=VALUE lines from .env (comments and empty lines skipped)
for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do set "%%A=%%B"
set "DATA_DIR=%~dp0data"
set "RESULTS_DIR=%~dp0results"
if "%~1"=="" (.venv\Scripts\python -m smcbot backtest) else (.venv\Scripts\python -m smcbot %*)
pause
