# Re-audit chain A (resumed): remaining machine-learning experiments, re-run from scratch (originals backed up in C:\bbbrep\backup_prerun_*). ad_ache_baselines already finished. Log: logs\ad_reaudit_A.log
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
Add-Type -Namespace Win -Name Pwr -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win.Pwr]::SetThreadExecutionState([uint32]2147483649)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: this process keeps the machine awake while it runs; released when it exits
$env:OPENBLAS_NUM_THREADS = "1"; $env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$py = "python"; $log = "logs\ad_reaudit_A.log"
foreach ($s in "ad_pgp_gate_validation","ad_random_vs_scaffold","ad_mmp_ablation","ad_split_deterministic_same_models","ad_paired_oracle_value","ad_v2_independent_checks") {
  "=== $s started $(Get-Date)" | Out-File $log -Append -Encoding utf8
  & $py -u -W ignore "src\$s.py" *> "logs\ad_reaudit_$s.log"
  "=== $s finished $(Get-Date) exit $LASTEXITCODE" | Out-File $log -Append -Encoding utf8
}
"=== CHAIN A DONE $(Get-Date)" | Out-File $log -Append -Encoding utf8
