@echo off
title Options Engine - GitHub Auto Backup
color 0A

set LOGFILE=C:\Users\leeha\.claude\projects\options-engine\backup_log.txt

echo ============================================
echo   OPTIONS ENGINE - GitHub Auto Backup
echo ============================================
echo.

:: Navigate to project folder
cd /d "C:\Users\leeha\.claude\projects\options-engine"

:: Check if git repo exists
if not exist ".git" (
    echo ERROR: No git repo found!
    echo.
    echo Fix: Run these commands:
    echo   git init
    echo   git remote add origin https://github.com/YOUR_USERNAME/options-engine.git
    echo.
    pause
    exit /b 1
)

:: Check if remote is configured
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: GitHub not connected yet!
    echo.
    echo Fix: Run this command:
    echo   git remote add origin https://github.com/YOUR_USERNAME/options-engine.git
    echo.
    pause
    exit /b 1
)

:: Check for changes
echo Checking for changes...
git status --short
echo.

git diff --quiet
git diff --cached --quiet
git status --porcelain > nul 2>&1

for /f "delims=" %%i in ('git status --porcelain') do (
    set HASCHANGES=1
)

if not defined HASCHANGES (
    echo No changes detected - already up to date!
    echo.
    echo Last commit:
    git log --oneline -1
    echo.
    echo SKIPPED - No changes >> "%LOGFILE%"
    goto SHOWLOG
)

:: Stage all files
echo Staging files...
git add .

:: Commit with simple message
git commit -m "Auto-backup"

:: Push to GitHub
echo.
echo Pushing to GitHub...
echo.
git push origin main

if %ERRORLEVEL% == 0 (
    echo.
    echo ============================================
    echo   SUCCESS - Backed up to GitHub!
    echo ============================================
    echo.
    git log --oneline -1
    echo.
    echo SUCCESS - Pushed to GitHub >> "%LOGFILE%"
) else (
    echo.
    echo ============================================
    echo   ERROR - Push Failed!
    echo ============================================
    echo.
    echo Most likely: GitHub credentials expired.
    echo.
    echo Fix: Run this manually:
    echo   git push origin main
    echo.
    echo ERROR - Push failed >> "%LOGFILE%"
)

:SHOWLOG
echo.
echo ============================================
echo   Recent backup history:
echo ============================================
if exist "%LOGFILE%" (
    powershell -command "Get-Content '%LOGFILE%' | Select-Object -Last 5"
) else (
    echo   No history yet - this is your first run!
)
echo ============================================
echo.
echo Closing in 15 seconds...
timeout /t 15
