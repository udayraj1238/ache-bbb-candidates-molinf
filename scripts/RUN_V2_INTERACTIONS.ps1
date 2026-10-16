Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
& "python" -u -W ignore src\ad_v2_interactions.py *> logs\ad_v2_interactions.log
