# Docks the property-matched decoy sets (840 molecules) for the docking-enrichment re-test.
# Launched through Task Scheduler, not Start-Process from an agent session (a session-owned process dies when the host application updates).
#   powershell -ExecutionPolicy Bypass -File .\RUN_DOCK_MATCHED.ps1
Set-Location "<PROJECT_ROOT>"
$py = "python"   # the environment with Meeko (the default Python has none)
if (-not (Test-Path $py)) { $py = ".\.venv\Scripts\python.exe" }
& $py -u src\ad_dock_enrichment_matched.py --workers 5 *> logs\ad_dock_matched.log


