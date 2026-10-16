# Deeper retrosynthesis pass (1000 iterations, 300 s) on molecules with no first-pass route, best margin first. One process (RAM).
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
& "C:\retro\venv\Scripts\python.exe" -u src\ad_v2_retro.py --in results\AD_V2_RETRO2_TODO.csv --out results\AD_V2_RETRO2.csv --config C:/retro/config_deep.yml *> logs\ad_v2_retro2.log
