# Second docking batch (stereo-resolved SMILES missing from the first list). Waits for V2Dock to finish, then docks. Low priority.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
while (-not (Select-String -Path logs\ad_v2_dock.log -Pattern 'done ->' -Quiet)) { Start-Sleep 30 }
$py = "python"
& $py -u src\ad_v2_dock.py --in results\AD_V2_TODOCK2.csv --workers 5 *> logs\ad_v2_dock2.log
