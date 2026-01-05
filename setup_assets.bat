@echo off
ECHO =======================================
ECHO  AUTOMATED ASSET SETUP SCRIPT
ECHO =======================================

REM Set the base directory to where this script is run
SET "BASE_DIR=%~dp0"
SET "ASSET_DIR=%BASE_DIR%assets"
SET "UI_DIR=%BASE_DIR%Osteovision ui\Fracture ui"

ECHO Creating 'assets' folder...
IF NOT EXIST "%ASSET_DIR%" (
    mkdir "%ASSET_DIR%"
    ECHO Folder 'assets' created.
) ELSE (
    ECHO Folder 'assets' already exists.
)

ECHO.
ECHO Copying AI Models (This may take a moment)...

IF EXIST "%BASE_DIR%vit_retrained.pth" (
    copy "%BASE_DIR%vit_retrained.pth" "%ASSET_DIR%\vit_retrained.pth" > NUL
    ECHO - vit_retrained.pth copied.
) ELSE (
    ECHO - WARNING: vit_retrained.pth not found.
)

IF EXIST "%BASE_DIR%swin_retrained.pth" (
    copy "%BASE_DIR%swin_retrained.pth" "%ASSET_DIR%\swin_retrained.pth" > NUL
    ECHO - swin_retrained.pth copied.
) ELSE (
    ECHO - WARNING: swin_retrained.pth not found.
)

IF EXIST "%BASE_DIR%monai_fracatlas_model.pth" (
    copy "%BASE_DIR%monai_fracatlas_model.pth" "%ASSET_DIR%\monai_fracatlas_model.pth" > NUL
    ECHO - monai_fracatlas_model.pth copied.
) ELSE (
    ECHO - WARNING: monai_fracatlas_model.pth not found.
)

IF EXIST "%BASE_DIR%Yolov11.pt" (
    copy "%BASE_DIR%Yolov11.pt" "%ASSET_DIR%\Yolov11.pt" > NUL
    ECHO - Yolov11.pt copied.
) ELSE (
    ECHO - WARNING: Yolov11.pt not found.
)

ECHO.
ECHO Copying Logo...
IF EXIST "%UI_DIR%\logo.jpg" (
    copy "%UI_DIR%\logo.jpg" "%ASSET_DIR%\logo.jpg" > NUL
    ECHO - logo.jpg copied.
) ELSE (
    ECHO - WARNING: 'logo.jpg' not found in %UI_DIR%
)

ECHO.
ECHO =======================================
ECHO  SETUP COMPLETE!
ECHO  The 'assets' folder is ready.
ECHO =======================================
ECHO.
ECHO You can now run Step 2:
ECHO streamlit run final_app.py
ECHO.
pause