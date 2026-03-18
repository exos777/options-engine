@echo off
title Options Engine - Workspace Launcher
color 0A

echo ============================================
echo   OPTIONS ENGINE - Workspace Launcher
echo ============================================
echo.

:: Open VS Code in project folder
echo Opening VS Code...
code "C:\Users\leeha\.claude\projects\options-engine"

echo.
echo ============================================
echo   VS Code is opening!
echo ============================================
echo.
echo Once VS Code opens:
echo.
echo   Press Ctrl+Shift+P
echo   Type: "Tasks: Run Task"
echo   Select: "Launch Full Workspace"
echo.
echo   This opens all 3 terminals automatically!
echo.
echo Or use keyboard shortcut after setup:
echo   Ctrl+Shift+B = Launch Full Workspace
echo.
timeout /t 8
