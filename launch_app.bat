@echo off
REM VrDeep - launch the desktop app.
REM
REM This hands straight over to launch_app.vbs and exits, so the console it was
REM started in closes immediately instead of sitting behind the app for the
REM whole session. Double-clicking launch_app.vbs directly avoids even the
REM moment this window is on screen.
REM
REM If the app does not appear, run launch_debug.bat - it keeps a console open
REM and shows what went wrong.

start "" wscript.exe "%~dp0launch_app.vbs"
