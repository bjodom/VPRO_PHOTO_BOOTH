# Test Guide

## Quick run

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
. .\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pytest tests\kiosk_app_test.py tests\kiosk_session_test.py tests\kiosk_flow_test.py tests\performance_recommendations_test.py tests\kiosk_framing_test.py tests\juggernaut_runner_test.py tests\juggernaut_test.py -q
```

This is the default hardware-free regression gate for the repo.

## 1) Activate the virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
. .\.venv\Scripts\Activate.ps1
```

## 2) Install pytest if needed

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
```

If the install hangs or fails on a restricted network, run the repo proxy helper from the parent folder first, then retry:

```powershell
cd ..
. .\proxy.ps1 -Enable
cd vPRO_Photo_Booth
.\.venv\Scripts\python.exe -m pip install pytest
```

## 3) Run the full hardware-free regression suite

```powershell
.\.venv\Scripts\python.exe -m pytest tests\kiosk_app_test.py tests\kiosk_session_test.py tests\kiosk_flow_test.py tests\performance_recommendations_test.py tests\kiosk_framing_test.py tests\juggernaut_runner_test.py tests\juggernaut_test.py -q
```

This is the default validation command used by the repo and is the current hardware-free gate.

## 4) Run a subset while debugging

```powershell
.\.venv\Scripts\python.exe -m pytest tests\kiosk_app_test.py -q
.\.venv\Scripts\python.exe -m pytest tests\kiosk_flow_test.py tests\kiosk_session_test.py -q
```

## 5) Run the entire test directory

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 6) Notes

- The project uses pytest fixtures such as `tmp_path`, and older tests may also request a legacy `tmp` fixture. A compatibility fixture is provided in [tests/conftest.py](conftest.py) so the suite remains runnable with current pytest versions.
- The current environment passed the full regression command with `59 passed, 1 warning`.
- The warning is a FastAPI/Starlette deprecation warning and does not fail the suite.
