# Stereochemistry resolution of the 159-molecule robust pool (docks isomers where needed). Low priority; Task Scheduler launch.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:OPENBLAS_NUM_THREADS = "2"
$py = "python"
& $py -u src\ad_resolve_stereo.py --in results\AD_V2_POOL.csv *> logs\ad_v2_stereo.log
