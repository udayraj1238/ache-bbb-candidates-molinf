# Retraining checks requested by the reviewer points (tautomers, descriptor filtering, like-for-like split protocol).
# Run one after another at low priority; launched through Task Scheduler (a session-owned process dies when the host application updates).
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$py = "python"
foreach ($s in @("ad_tautomer_effect","ad_filter_sensitivity","ad_split_deterministic_same_models")) {
    "=== $s started $(Get-Date)" | Out-File logs\ad_checks.log -Append -Encoding utf8
    & $py -u "src\$s.py" *>> logs\ad_checks.log
    "=== $s finished $(Get-Date) exit $LASTEXITCODE" | Out-File logs\ad_checks.log -Append -Encoding utf8
}
