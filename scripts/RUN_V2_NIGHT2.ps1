# Routes export (iteration-limited, so CPU load cannot change the result), then rebuild paper, dossier and Word. Log: logs\ad_v2_night.log
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$log = "logs\ad_v2_night.log"; function Say($m) { "=== $m $(Get-Date)" | Out-File $log -Append -Encoding utf8 }
Say "routes export started (v2)"
& "C:\retro\venv\Scripts\python.exe" -u src\ad_v2_routes.py *> logs\ad_v2_routes.log
Say "routes exported exit $LASTEXITCODE"
& "python" -W ignore src\ad_v2_finalize_paper.py *>> $log
Say "ALL DONE (stage 2)"
