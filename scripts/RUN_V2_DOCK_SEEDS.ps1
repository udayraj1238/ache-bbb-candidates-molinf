# Docking reproducibility: re-dock the delivered set from two new seeds. 4 workers, low priority. Log: logs\ad_v2_dock_seeds.log
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
& "python" -u -W ignore src\ad_v2_dock_seeds.py *> logs\ad_v2_dock_seeds.log
