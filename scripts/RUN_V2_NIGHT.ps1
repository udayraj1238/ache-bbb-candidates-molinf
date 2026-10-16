# Overnight chain: (1) wait for the retrosynthesis benchmark and the GNN check, fold both into the paper; (2) export step-by-step routes; (3) rebuild paper, dossier and Word file. Log: logs\ad_v2_night.log
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$log = "logs\ad_v2_night.log"; function Say($m) { "=== $m $(Get-Date)" | Out-File $log -Append -Encoding utf8 }
$py = "python"
while (-not (Select-String -Path logs\ad_v2_retro_bench.log -Pattern 'done ->' -Quiet -ErrorAction SilentlyContinue)) { Start-Sleep 30 }
Say "benchmark finished"
while (-not (Test-Path results\AD_V2_GNN_SUMMARY.txt)) { Start-Sleep 30; if (-not (Get-Process python -ErrorAction SilentlyContinue | ? { $_.Id -ne $PID })) { break } }
Say "gnn summary present: $(Test-Path results\AD_V2_GNN_SUMMARY.txt)"
& $py -W ignore src\ad_v2_finalize_paper.py *>> $log
Say "paper finalized (stage 1)"
& "C:\retro\venv\Scripts\python.exe" -u src\ad_v2_routes.py *> logs\ad_v2_routes.log
Say "routes exported exit $LASTEXITCODE"
& $py -W ignore src\ad_v2_finalize_paper.py *>> $log
Say "ALL DONE (stage 2)"
