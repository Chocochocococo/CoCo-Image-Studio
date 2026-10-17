@echo off
chcp 65001 >nul
title CoCo Image Studio
rem This file lives in data\ with everything else CoCo uses: the program, its
rem Python and packages, the images, models and settings. Next to data\ there
rem is only "CoCo Image Studio.exe" (it runs this file and carries the CoCo
rem icon, which a .bat cannot have) plus the README and the license. Running
rem this file directly works the same. Every path below is relative to data\.
cd /d "%~dp0"
set "COCO_DIR=%~dp0"
set "COCO_BAT=%~f0"
for %%F in ("%~dp0..") do set "COCO_TOP=%%~fF\"
set "COCO_EXE=%COCO_TOP%CoCo Image Studio.exe"
rem What shortcuts open, and their icon: the exe when it is there.
set "COCO_OPEN=%COCO_EXE%"
set "COCO_ICON=%COCO_EXE%,0"
if not exist "%COCO_EXE%" set "COCO_OPEN=%COCO_BAT%"
if not exist "%COCO_EXE%" set "COCO_ICON=%COCO_DIR%assets\coco.ico"

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

rem CoCo's own Python (python\), downloaded on the first setup from the
rem official python.org package on nuget.org and checked against the SHA512
rem that nuget.org publishes. Every new install gets the Python the lock file
rem is tested with, and a Python already on the computer is left alone.
rem Without a connection it falls back to an installed Python 3.11-3.13.
set "PY_VER=3.11.9"
set "PY_EXE=python\python.exe"
set "PY_URL=https://api.nuget.org/v3-flatcontainer/python/%PY_VER%/python.%PY_VER%.nupkg"
set "PY_SHA512=e353a7efd159ef98ab9ce1011854a30d5d778f59d96fc9bb11b07ced2990b355f030486b0707f798c73c4420173ab111876a96ead5981718b604c4cdd71c558d"

rem Messages. English is set here; when CoCo speaks Chinese (picked in its
rem settings, or the Windows display language - see LANG below) the lines in
rem web\locales\launcher.zh-TW.txt replace them. That file is UTF-8 (chcp
rem 65001 above) so this one can stay plain ASCII. %%NAME%% stays in the text
rem as a placeholder; the lines that use one print with "call echo".
call :LANG
set "M_PICKTITLE=CoCo Image Studio - Choose an install"
set "M_PICKFULL=1  Full   (about 4 GB, needs an NVIDIA GPU)"
set "M_PICKFULL2=Everything: generate, refine and train, plus all of Light."
set "M_PICKLITE=2  Light  (about 1 GB)"
set "M_PICKLITE2=Gallery, explore, compare, recombine, output, the library, and prompts and tags."
set "M_PICKLITE3=Explore runs on the CPU. You can switch to Full later in Settings."
set "M_NOGPU=No NVIDIA GPU was found on this computer, so Light is recommended."
set "M_PICKASK=Type 1 or 2 (recommended: %%PICK%%): "
set "M_CHECKPKG=[*] Checking CoCo packages..."
set "M_DEPSWARN1=[!] Could not update CoCo's packages (offline?). Starting with what is installed."
set "M_DEPSWARN2=Features that need the new packages (such as the Beta schedule) stay off"
set "M_DEPSWARN3=until a later start can reach the internet."
set "M_NOCUDA1=[!] PyTorch %%TORCH_VER%%+%%CU_TAG%% is installed, but it cannot see your GPU."
set "M_NOCUDA2=Reinstalling will not help - this is a driver or hardware problem."
set "M_NOCUDA3=Check the NVIDIA driver (CUDA 13 needs 580 or newer)."
set "M_NOCUDA4=The gallery still works; image generation will report the same error."
set "M_SETUPTITLE=CoCo Image Studio - Setup"
set "M_FINDPY=[*] Looking for Python 3.11, 3.12, or 3.13 on this computer instead..."
set "M_FOUNDPY=Found Python"
set "M_MKVENV=[*] Creating virtual environment..."
set "M_PIPUP=[*] Updating pip..."
set "M_TORCHCUDA=[*] Installing PyTorch %%TORCH_VER%% with CUDA 13.0..."
set "M_DEPS=[*] Installing CoCo dependencies..."
set "M_VERIFYCUDA=[*] Verifying CUDA environment..."
set "M_TORCHCPU=[*] Installing PyTorch %%TORCH_VER%% (CPU only)..."
set "M_VERIFY=[*] Verifying..."
set "M_STARTING=[*] Scanning the gallery and starting Studio..."
set "M_GETPY=[*] Downloading Python %%PY_VER%% from python.org (about 17 MB)..."
set "M_BADSUM=[!] The download does not match the official checksum, so it is not used."
set "M_PYREADY=Python %%PY_VER%% is ready."
set "M_PYFAIL=[!] Could not download Python %%PY_VER%%."
set "M_SCASK=Create a CoCo shortcut on the desktop and in the Start menu? (Y/N): "
set "M_SCFAIL=[!] Could not create the shortcut."
set "M_SCDONE=Shortcut created: CoCo Image Studio"
set "M_NOPY1=[X] Could not download Python, and Python 3.11, 3.12, or 3.13 was not found"
set "M_NOPY2=on this computer. Check the internet connection, then run CoCo.bat again."
set "M_VENVFAIL=[X] Failed to create the virtual environment."
set "M_PIPFAIL=[X] Package installation failed. See the messages above."
set "M_VERIFYFAIL=[X] Installation finished, but the final check failed. See the messages above."
set "M_STUDIOFAIL=[X] Studio stopped with an error. (exit code %%RC%%)"
set "M_MISSING=[X] Required project files are missing. Extract the complete project first."
set "M_MOVEFAIL1=[X] Could not move the old files into the data folder. CoCo may still be open."
set "M_MOVEFAIL2=Close every CoCo window, then run CoCo.bat again."
set "MSG_ZH=web\locales\launcher.zh-TW.txt"
if "%UILANG%"=="zh" if exist "%MSG_ZH%" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%MSG_ZH%") do set "M_%%A=%%B"

if not exist "tools\studio.py" goto MISSING
rem Only next to an installed CoCo (its exe, or the CoCo.bat of the old layout)
rem is the folder above ours CoCo's own: never take anything from elsewhere.
set "OLDTOP="
if exist "%COCO_EXE%" set "OLDTOP=1"
if exist "%COCO_TOP%CoCo.bat" set "OLDTOP=1"
if defined OLDTOP call :RELOCATE || goto MOVEFAIL
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
if exist "venv\.coco-python" if exist "%PY_EXE%" call :FIXVENV
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
echo   %M_PICKTITLE%
echo ============================================
echo.
echo   %M_PICKFULL%
echo      %M_PICKFULL2%
echo.
echo   %M_PICKLITE%
echo      %M_PICKLITE2%
echo      %M_PICKLITE3%
echo.
set "PICK=1"
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    set "PICK=2"
    echo   %M_NOGPU%
    echo.
)
call set "ASKMSG=%M_PICKASK%"
choice /c 12 /n /m "%ASKMSG%"
if errorlevel 2 (set "MODE=lite") else set "MODE=full"
>"%MODE_FILE%" echo %MODE%
rem Ask for the shortcut now, while someone is reading this window. Asking
rem only after the install, with a timeout, was always missed: by then people
rem watch for the browser. The answer is used once the venv exists (SHORTCUT).
echo.
choice /c YN /n /m "%M_SCASK%"
if errorlevel 2 (set "SCWANT=N") else set "SCWANT=Y"
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
echo %M_CHECKPKG%
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
echo %M_CHECKPKG%
"%VENV_PY%" -m pip install -r requirements-lite.txt -c requirements-lock.txt -q
if errorlevel 1 goto DEPSWARN
"%VENV_PY%" -c "import torch, transformers, safetensors, truststore, PIL, numpy" >nul 2>nul
if errorlevel 1 goto SETUP
>"%READY_FILE%" echo %READY_TAG%
goto START

:DEPSWARN
rem No marker is written, so the next start tries again.
echo.
echo %M_DEPSWARN1%
echo     %M_DEPSWARN2%
echo     %M_DEPSWARN3%
echo.
goto START

:NOCUDA
echo.
call echo %M_NOCUDA1%
echo     %M_NOCUDA2%
echo     %M_NOCUDA3%
echo     %M_NOCUDA4%
echo.
goto START

:SETUP
echo.
echo ============================================
echo   %M_SETUPTITLE%
echo ============================================
echo.

if exist "%VENV_PY%" goto INSTALL

if not exist "%PY_EXE%" call :GETPY
if exist "%PY_EXE%" goto MKVENV

echo %M_FINDPY%
set "PYEXE="
for %%V in (3.12 3.11 3.13) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            set "PYEXE=py -%%V"
            echo     %M_FOUNDPY% %%V
        )
    )
)
if not defined PYEXE goto NOPY

echo.
echo %M_MKVENV%
%PYEXE% -m venv venv
if errorlevel 1 goto VENVFAIL
goto INSTALL

:MKVENV
echo.
echo %M_MKVENV%
"%PY_EXE%" -m venv venv
if errorlevel 1 goto VENVFAIL
>"venv\.coco-python" echo %PY_VER%

:INSTALL
echo.
echo %M_PIPUP%
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
call echo %M_TORCHCUDA%
"%VENV_PY%" -m pip uninstall -y torch torchvision >nul 2>nul
"%VENV_PY%" -m pip install torch==%TORCH_VER%+%CU_TAG% torchvision==%TORCHVISION_VER%+%CU_TAG% --index-url https://download.pytorch.org/whl/%CU_TAG%
if errorlevel 1 goto PIPFAIL

echo.
echo %M_DEPS%
"%VENV_PY%" -m pip install -r requirements-lock.txt
if errorlevel 1 goto PIPFAIL

echo.
echo %M_VERIFYCUDA%
"%VENV_PY%" -c "import torch,diffusers; assert torch.cuda.is_available(), 'CUDA is not available'; print('  torch',torch.__version__); print('  diffusers',diffusers.__version__); print('  GPU:',torch.cuda.get_device_name(0)); print('  capability:',torch.cuda.get_device_capability(0))"
if errorlevel 1 goto VERIFYFAIL
>"%READY_FILE%" echo %READY_TAG%
goto START

:INSTALLLITE
echo.
call echo %M_TORCHCPU%
"%VENV_PY%" -m pip uninstall -y torch torchvision >nul 2>nul
"%VENV_PY%" -m pip install torch==%TORCH_VER%+cpu --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto PIPFAIL

echo.
echo %M_DEPS%
"%VENV_PY%" -m pip install -r requirements-lite.txt -c requirements-lock.txt
if errorlevel 1 goto PIPFAIL

echo.
echo %M_VERIFY%
"%VENV_PY%" -c "import torch, transformers, safetensors, truststore, PIL, numpy; print('  torch', torch.__version__, '(CPU)'); print('  transformers', transformers.__version__)"
if errorlevel 1 goto VERIFYFAIL
>"%READY_FILE%" echo %READY_TAG%

:START
call :SHORTCUT
echo.
echo %M_STARTING%
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

:RELOCATE
rem Until 2026-10-16 everything sat next to CoCo.bat. The program moves its
rem own files into data\ on its first start (server\relocate.py), but the
rem venv, the Python it runs from and the install choice must be in place
rem before it starts, so they are moved here. A folder in use (CoCo still
rem open) cannot be moved: stop and say so, instead of installing a second
rem copy. If data\ already has one, the old one is left where it is.
set "MOVED="
set "MOVEBAD="
for %%D in (venv python) do (
    if exist "..\%%D\" if not exist "%%D\" (
        move "..\%%D" "%%D" >nul 2>nul
        if errorlevel 1 (set "MOVEBAD=%%D") else set "MOVED=1"
    )
)
if exist "..\install_mode.txt" if not exist "install_mode.txt" move "..\install_mode.txt" "install_mode.txt" >nul 2>nul
if defined MOVEBAD exit /b 1
if defined MOVED call :FIXLINKS
exit /b 0

:FIXLINKS
rem Shortcuts made before the move open the old CoCo.bat and take their icon
rem from the old place. Only the ones that open this CoCo (its exe, this file
rem or the old CoCo.bat next to data\) are touched; they now open the exe.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$w = New-Object -ComObject WScript.Shell; $mine = @($env:COCO_OPEN, $env:COCO_BAT, ($env:COCO_TOP + 'CoCo.bat')); foreach ($d in [Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs')) { $p = Join-Path $d 'CoCo Image Studio.lnk'; if (Test-Path $p) { $s = $w.CreateShortcut($p); if ($mine -contains $s.TargetPath) { $s.TargetPath = $env:COCO_OPEN; $s.WorkingDirectory = $env:COCO_DIR; $s.IconLocation = $env:COCO_ICON; $s.Save() } } }" >nul 2>nul
exit /b 0

:LANG
rem The language the launcher speaks, picked the same way as the server's
rem console messages (server\app.py, startup_lang): the language chosen in
rem CoCo's settings (gallery_ui.json), otherwise the Windows display
rem language. That one is read from the registry; PowerShell can report a
rem different culture than the one the desktop shows.
set "UILANG=en"
set "UI_JSON=gallery_ui.json"
if not exist "%UI_JSON%" if exist "..\gallery_ui.json" set "UI_JSON=..\gallery_ui.json"
if exist "%UI_JSON%" findstr /c:"\"lang\": \"zh-TW\"" "%UI_JSON%" >nul 2>nul && set "UILANG=zh" && exit /b 0
if exist "%UI_JSON%" findstr /c:"\"lang\": \"en\"" "%UI_JSON%" >nul 2>nul && exit /b 0
set "WINLANG="
for /f "tokens=3" %%L in ('reg query "HKCU\Control Panel\Desktop" /v PreferredUILanguages 2^>nul ^| findstr /i "REG_MULTI_SZ"') do set "WINLANG=%%L"
if not defined WINLANG for /f "tokens=3" %%L in ('reg query "HKCU\Control Panel\Desktop\MuiCached" /v MachinePreferredUILanguages 2^>nul ^| findstr /i "REG_MULTI_SZ"') do set "WINLANG=%%L"
if defined WINLANG if /i "%WINLANG:~0,3%"=="zh-" set "UILANG=zh"
exit /b 0

:GETPY
echo.
call echo %M_GETPY%
if exist "python.tmp" rmdir /s /q "python.tmp"
"%SystemRoot%\System32\curl.exe" -L --fail --silent --show-error -o "python.nupkg" "%PY_URL%"
if errorlevel 1 goto GETPYBAD
certutil -hashfile "python.nupkg" SHA512 | findstr /i /x "%PY_SHA512%" >nul
if errorlevel 1 (
    echo %M_BADSUM%
    goto GETPYBAD
)
mkdir "python.tmp"
"%SystemRoot%\System32\tar.exe" -xf "python.nupkg" -C "python.tmp" tools
if errorlevel 1 goto GETPYBAD
move "python.tmp\tools" "python" >nul
if errorlevel 1 goto GETPYBAD
rmdir /s /q "python.tmp"
del "python.nupkg"
call echo     %M_PYREADY%
exit /b 0

:GETPYBAD
if exist "python.nupkg" del "python.nupkg"
if exist "python.tmp" rmdir /s /q "python.tmp"
call echo %M_PYFAIL%
exit /b 1

:FIXVENV
rem The venv was made from CoCo's own Python. pyvenv.cfg holds that Python's
rem full path, so it is written again on every start: the CoCo folder can be
rem moved or renamed without reinstalling.
>"venv\pyvenv.cfg" echo home = %COCO_DIR%python
>>"venv\pyvenv.cfg" echo include-system-site-packages = false
>>"venv\pyvenv.cfg" echo version = %PY_VER%
exit /b 0

:SHORTCUT
rem Once per venv (the marker lives in venv\): a shortcut that opens the exe,
rem with its icon, on the desktop and in the Start menu. A new install asked
rem when the install mode was picked (SCWANT). Installs made before that, or
rem an install picked up again later, are asked here; no answer in 60 seconds
rem means no, so a start nobody watches is not held up. The paths go through
rem environment variables, not the command line.
if exist "venv\.coco-shortcut" exit /b 0
>"venv\.coco-shortcut" echo asked
if not defined SCWANT call :SCASK
if /i not "%SCWANT%"=="Y" exit /b 0
powershell -NoProfile -ExecutionPolicy Bypass -Command "$w = New-Object -ComObject WScript.Shell; foreach ($d in [Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs')) { $s = $w.CreateShortcut((Join-Path $d 'CoCo Image Studio.lnk')); $s.TargetPath = $env:COCO_OPEN; $s.WorkingDirectory = $env:COCO_DIR; $s.IconLocation = $env:COCO_ICON; $s.Description = 'CoCo Image Studio'; $s.Save() }" >nul 2>nul
if errorlevel 1 (echo %M_SCFAIL%) else echo     %M_SCDONE%
exit /b 0

:SCASK
echo.
choice /c YN /n /t 60 /d N /m "%M_SCASK%"
if errorlevel 2 (set "SCWANT=N") else set "SCWANT=Y"
exit /b 0

:NOPY
echo.
echo %M_NOPY1%
echo     %M_NOPY2%
goto FAIL

:VENVFAIL
echo.
echo %M_VENVFAIL%
goto FAIL

:PIPFAIL
echo.
echo %M_PIPFAIL%
goto FAIL

:VERIFYFAIL
echo.
echo %M_VERIFYFAIL%
goto FAIL

:STUDIOFAIL
echo.
call echo %M_STUDIOFAIL%
goto FAIL

:MOVEFAIL
echo.
echo %M_MOVEFAIL1%
echo     %M_MOVEFAIL2%
goto FAIL

:MISSING
echo.
echo %M_MISSING%

:FAIL
echo.
pause
exit /b 1
