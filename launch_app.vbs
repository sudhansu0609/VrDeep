' VrDeep - launch the desktop app with no console window.
'
' Double-clicking a .bat always creates a console, and python.exe keeps one for
' as long as the app runs - which is why starting the app used to put two
' windows on screen. Two things fix that between them: pythonw.exe is python
' without a console, and wscript (the default handler for a .vbs) is not a
' console program itself, so nothing flashes on the way through either.
'
' Anything the app would have printed goes to output\app.log instead. Use
' launch_debug.bat when you want to watch that as it happens.

Option Explicit

Dim fso, shell, projectDir, pythonw, appPy, torchLib, trtLib, env

Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

projectDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw = projectDir & "\facefusion\venv\Scripts\pythonw.exe"
appPy = projectDir & "\scripts\app.py"

' With no console there is nowhere to print a startup problem, so the two that
' stop the app existing at all get a dialog of their own.
If Not fso.FileExists(pythonw) Then
    MsgBox "Python was not found at:" & vbCrLf & vbCrLf & pythonw & vbCrLf & vbCrLf & _
           "Make sure FaceFusion is set up in the 'facefusion' folder.", _
           vbCritical, "VrDeep"
    WScript.Quit 1
End If

If Not fso.FileExists(appPy) Then
    MsgBox "app.py was not found at:" & vbCrLf & vbCrLf & appPy, vbCritical, "VrDeep"
    WScript.Quit 1
End If

' The same CUDA and TensorRT DLL directories the batch launcher put on PATH.
' Without them the GPU providers fall back to CPU, which is the difference
' between minutes and hours. Changing PROCESS environment here is enough - the
' app is started from this process and inherits it.
torchLib = projectDir & "\facefusion\venv\Lib\site-packages\torch\lib"
trtLib = projectDir & "\facefusion\venv\Lib\site-packages\tensorrt_libs"
Set env = shell.Environment("PROCESS")
If fso.FolderExists(torchLib) Then env("PATH") = torchLib & ";" & env("PATH")
If fso.FolderExists(trtLib) Then env("PATH") = trtLib & ";" & env("PATH")

' 0 = hidden window, False = do not wait for it to exit.
shell.Run """" & pythonw & """ """ & appPy & """", 0, False
