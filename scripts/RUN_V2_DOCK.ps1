# Vina score and pose coverage for the synthesis-ready selection pool. Low priority, 5 workers; Task Scheduler launch.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$py = "python"
& $py -u src\ad_v2_dock.py --in results\AD_V2_TODOCK.csv --workers 5 *> logs\ad_v2_dock.log
