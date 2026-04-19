@echo off
powershell -NoExit -Command "cd 'C:\Users\leeha\.claude\projects\options-engine'; .venv\Scripts\python.exe -m streamlit run app/main.py"