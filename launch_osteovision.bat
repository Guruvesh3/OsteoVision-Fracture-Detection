@echo off
ECHO Launching OsteoVision...
ECHO This window must remain open while the app is running.

REM We are bypassing 'conda activate' and calling the environment's streamlit.exe directly
REM This path is based on your 'cpu_env' environment.

ECHO Launching Streamlit from cpu_env...
C:\Users\Guruvesh\miniconda3\envs\cpu_env\Scripts\streamlit.exe run "C:\Users\Guruvesh\Downloads\extracted_files\final_app.py"

pause