# Re-audit chain B (resumed): docking experiments re-run from scratch. Log: logs\ad_reaudit_B.log
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
Add-Type -Namespace Win -Name Pwr -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Pwr]::SetThreadExecutionState([uint32]2147483649)
$env:OPENBLAS_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"   # one thread per worker: far less committed memory per process
$py = "python"; $log = "logs\ad_reaudit_B.log"
"=== enrichment started $(Get-Date)" | Out-File $log -Append -Encoding utf8
& $py -u -W ignore src\ad_dock_enrichment_matched.py --workers 6 *> logs\ad_reaudit_dock_enrichment.log
"=== enrichment finished $(Get-Date) exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
"=== CHAIN B DONE $(Get-Date)" | Out-File $log -Append -Encoding utf8

