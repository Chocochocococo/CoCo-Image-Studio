@echo off
chcp 65001 >nul
cd /d "%~dp0"
title CoCo Image Studio

set "VENV_PY=venv\Scripts\python.exe"
set "READY_FILE=venv\.coco-ready"
rem Full or Light install, asked on the first setup (ASK below) and kept in
rem MODE_FILE. Light has no generation engine: CPU-only PyTorch plus
rem requirements-lite.txt, for the gallery, explore, compare, recombine, output
rem and the library. Settings can switch it to Full (server\install.py writes
rem "full" here); the next start then installs the rest. Installs made before
rem this choice existed are Full.
set "MODE_FILE=install_mode.txt"

rem The pinned CUDA build. Used by both the installer and the probe below, so
rem that deleting the ready marker really does re-check what is installed.
set "TORCH_VER=2.14.0"
set "TORCHVISION_VER=0.29.0"
set "CU_TAG=cu130"
rem Bump DEPS_REV whenever requirements-lock.txt changes. The ready marker
rem records it, so an existing install syncs the CoCo packages on its next
rem start (PROBE below) without touching torch.
rem   2  scipy for the Beta schedule
rem   3  ultralytics for the face fix
rem   4  every package they depend on is pinned too (tools\lock_deps.py)
rem   5  ultralytics dropped (CoCo runs the face finder itself); opencv listed directly
rem   6  comfy-kitchen for running int8 models in int8 (Qwen Image 2.1)
rem   7  truststore: downloads use the Windows certificate store (antivirus HTTPS scanning)
set "DEPS_REV=7"
set "READY_TAG=%TORCH_VER%+%CU_TAG% deps%DEPS_REV%"
set "LITE_TAG=%TORCH_VER%+cpu deps%DEPS_REV% lite"

if not exist "tools\studio.py" goto MISSING
if not exist "requirements-lock.txt" goto MISSING
if not exist "requirements-lite.txt" goto MISSING
set "MODE="
if exist "%MODE_FILE%" set /p MODE=<"%MODE_FILE%"
if not defined MODE if exist "%VENV_PY%" (
    set "MODE=full"
    >"%MODE_FILE%" echo full
)
if not defined MODE goto ASK

:MODESET
if /i "%MODE%"=="lite" set "READY_TAG=%LITE_TAG%"
if not exist "%VENV_PY%" goto SETUP
rem The marker records which torch build and dependency revision it vouches
rem for. When either pin above changes (or the marker predates this and just
rem says "ready"), fall through to PROBE instead of starting on a stale setup.
set "READY="
if exist "%READY_FILE%" set /p READY=<"%READY_FILE%"
if "%READY%"=="%READY_TAG%" goto START
goto PROBE

:ASK
echo.
echo ============================================
echo   CoCo Image Studio - Choose an install
echo ============================================
echo.
echo   1  Full   (about 4 GB, needs an NVIDIA GPU)
echo      Everything: generate, refine and train, plus all of Light.
echo.
echo   2  Light  (about 1 GB)
echo      Gallery, explore, compare, recombine, output and the library.
echo      Explore runs on the CPU. You can switch to Full later in Settings.
echo.
set "PICK=1"
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    set "PICK=2"
    echo   No NVIDIA GPU was found on this computer, so Light is recommended.
    echo.
)
choice /c 12 /n /m "Type 1 or 2 (recommended: %PICK%): "
if errorlevel 2 (set "MODE=lite") else set "MODE=full"
>"%MODE_FILE%" echo %MODE%
goto MODESET

:PROBE
rem No matching ready marker, but a venv exists. This covers projects created
rem before the marker, an interrupted install, the documented repair step of
rem deleting the marker, a changed requirements-lock.txt (DEPS_REV), and a
rem Light install switched to Full (its torch is the CPU build -> SETUP).
rem   torch missing or the wrong build  -> full reinstall (SETUP)
rem Otherwise sync the CoCo packages with the lock file (quick when nothing
rem changed; torch is left alone), then check imports and CUDA:
rem   0  all good                       -> write marker and start
rem   2  right packages, no CUDA        -> driver or GPU problem;
rem                                        reinstalling cannot fix it
rem   1  imports still failing          -> full reinstall
if /i "%MODE%"=="lite" goto PROBELITE
"%VENV_PY%" -c "import sys, torch; sys.exit(0 if torch.__version__.startswith('%TORCH_VER%+%CU_TAG%') else 1)" >nul 2>nul
if errorlevel 1 goto SETUP
echo [*] Checking CoCo packages...
"%VENV_PY%" -m pip install -r requirements-lock.txt -q
if errorlevel 1 goto DEPSWARN
"%VENV_PY%" -c "import sys, torch, torchvision, diffusers, transformers, accelerate, safetensors, sentencepiece, google.protobuf, peft, PIL, scipy, cv2, pynvml, comfy_kitchen, truststore; sys.exit(0 if torch.cuda.is_available() else 2)" >nul 2>nul
if errorlevel 2 goto NOCUDA
if errorlevel 1 goto SETUP
>"%READY_FILE%" echo %READY_TAG%
goto START

:PROBELITE
rem Same as PROBE for a Light install: CPU torch, the lite package list with
rem the versions from requirements-lock.txt, no GPU check.
"%VENV_PY%" -c "import sys, torch; sys.exit(0 if torch.__version__.startswith('%TORCH_VER%+cpu') else 1)" >nul 2>nul
if errorlevel 1 goto SETUP
echo [*] Checking CoCo packages...
"%VENV_PY%" -m pip install -r requirements-lite.txt -c requirements-lock.txt -q
if errorlevel 1 goto DEPSWARN
"%VENV_PY%" -c "import torch, transformers, safetensors, truststore, PIL, numpy" >nul 2>nul
if errorlevel 1 goto SETUP
>"%READY_FILE%" echo %READY_TAG%
goto START

:DEPSWARN
rem No marker is written, so the next start tries again.
echo.
echo [!] Could not update CoCo's packages (offline?). Starting with what is installed.
echo     Features that need the new packages (such as the Beta schedule) stay off
echo     until a later start can reach the internet.
echo.
goto START

:NOCUDA
echo.
echo [!] PyTorch %TORCH_VER%+%CU_TAG% is installed, but it cannot see your GPU.
echo     Reinstalling will not help - this is a driver or hardware problem.
echo     Check the NVIDIA driver (CUDA 13 needs 580 or newer).
echo     The gallery still works; image generation will report the same error.
echo.
goto START

:SETUP
echo.
echo ============================================
echo   CoCo Image Studio - Setup
echo ============================================
echo.

if exist "%VENV_PY%" goto INSTALL

echo [*] Looking for Python 3.11, 3.12, or 3.13...
set "PYEXE="
for %%V in (3.12 3.11 3.13) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            set "PYEXE=py -%%V"
            echo     Found Python %%V
        )
    )
)
if not defined PYEXE goto NOPY

echo.
echo [*] Creating virtual environment...
%PYEXE% -m venv venv
if errorlevel 1 goto VENVFAIL

:INSTALL
echo.
echo [*] Updating pip...
rem The pip that comes with Python (24.0 with 3.11.9) only trusts its own
rem certificate list. Antivirus HTTPS scanning (Norton and others) signs
rem every site with its own certificate, so that pip cannot download
rem anything, not even its own update. truststore makes it use the Windows
rem certificate store; pip 24.2 and newer do that by default. Older pips
rem without truststore reject the option, so try once more without it.
"%VENV_PY%" -m pip install --upgrade pip -q --use-feature=truststore >nul 2>nul
if errorlevel 1 "%VENV_PY%" -m pip install --upgrade pip -q
if errorlevel 1 goto PIPFAIL
if /i "%MODE%"=="lite" goto INSTALLLITE

echo.
echo [*] Installing PyTorch %TORCH_VER% with CUDA 13.0...
"%VENV_PY%" -m pip uninstall -y torch torchvision >nul 2>nul
"%VENV_PY%" -m pip install torch==%TORCH_VER%+%CU_TAG% torchvision==%TORCHVISION_VER%+%CU_TAG% --index-url https://download.pytorch.org/whl/%CU_TAG%
if errorlevel 1 goto PIPFAIL

echo.
echo [*] Installing CoCo dependencies...
"%VENV_PY%" -m pip install -r requirements-lock.txt
if errorlevel 1 goto PIPFAIL

echo.
echo [*] Verifying CUDA environment...
"%VENV_PY%" -c "import torch,diffusers; assert torch.cuda.is_available(), 'CUDA is not available'; print('  torch',torch.__version__); print('  diffusers',diffusers.__version__); print('  GPU:',torch.cuda.get_device_name(0)); print('  capability:',torch.cuda.get_device_capability(0))"
if errorlevel 1 goto VERIFYFAIL
>"%READY_FILE%" echo %READY_TAG%
goto START

:INSTALLLITE
echo.
echo [*] Installing PyTorch %TORCH_VER% (CPU only)...
"%VENV_PY%" -m pip uninstall -y torch torchvision >nul 2>nul
"%VENV_PY%" -m pip install torch==%TORCH_VER%+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto PIPFAIL

echo.
echo [*] Installing CoCo dependencies...
"%VENV_PY%" -m pip install -r requirements-lite.txt -c requirements-lock.txt
if errorlevel 1 goto PIPFAIL

echo.
echo [*] Verifying...
"%VENV_PY%" -c "import torch, transformers, safetensors, truststore, PIL, numpy; print('  torch', torch.__version__, '(CPU)'); print('  transformers', transformers.__version__)"
if errorlevel 1 goto VERIFYFAIL
>"%READY_FILE%" echo %READY_TAG%

:START
echo.
echo [*] Scanning the gallery and starting Studio...
"%VENV_PY%" tools\studio.py
set "RC=%ERRORLEVEL%"
rem Do not use `if errorlevel 1`: it means "greater than or equal to 1", and
rem Windows crash codes are negative to cmd, so every one of them slips past
rem (access violation 0xC0000005 shows up as -1073741819).
rem Ctrl+C is the opposite case - STATUS_CONTROL_C_EXIT (0xC000013A, seen as
rem -1073741510) means the user closed it on purpose, which is not an error.
if "%RC%"=="0" exit /b 0
if "%RC%"=="-1073741510" exit /b 0
goto STUDIOFAIL

:NOPY
echo.
echo [X] Python 3.11, 3.12, or 3.13 was not found.
echo     Install Python 3.12 with the Python launcher, then run CoCo.bat again.
goto FAIL

:VENVFAIL
echo.
echo [X] Failed to create the virtual environment.
goto FAIL

:PIPFAIL
echo.
echo [X] Package installation failed. See the messages above.
goto FAIL

:VERIFYFAIL
echo.
echo [X] Installation finished, but the final check failed. See the messages above.
goto FAIL

:STUDIOFAIL
echo.
echo [X] Studio stopped with an error. (exit code %RC%)
goto FAIL

:MISSING
echo.
echo [X] Required project files are missing. Extract the complete project first.

:FAIL
echo.
pause
exit /b 1
