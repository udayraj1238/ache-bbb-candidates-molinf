# Shard 0 of the retrosynthesis, run after shard 1 finishes (RAM: each process holds the 17M-compound stock, ~2.3 GB; only ~2 GB is free alongside Chrome/VS Code).
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
while (-not (Select-String -Path logs\ad_v2_retro_s1.log -Pattern 'done ->' -Quiet)) { Start-Sleep 30 }
& "C:\retro\venv\Scripts\python.exe" -u src\ad_v2_retro.py --in results\AD_V2_OFFLINE.csv --only-pass --out results\AD_V2_RETRO_s0.csv --shard 0 --nshards 2 *> logs\ad_v2_retro_s0.log
