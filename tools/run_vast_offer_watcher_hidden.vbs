' Run the existing PowerShell supervisor without opening a console every minute.
' Keep the original runner and its durable scheduled-last.json receipt unchanged.
Dim mode, shell, fso, runnerPath, quote, command, exitCode
mode = "Supervise"
If WScript.Arguments.Count > 0 Then mode = WScript.Arguments(0)
If mode <> "Observe" And mode <> "Auto" And mode <> "Supervise" Then WScript.Quit 2
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
runnerPath = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), "run_vast_offer_watcher_task.ps1")
quote = Chr(34)
command = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " & quote & runnerPath & quote & " -Mode " & mode
exitCode = shell.Run(command, 0, True)
WScript.Quit exitCode
