# Final chain for the synthesis set. Waits for BOTH docking jobs, then: offline gates -> four-source novelty (sequential, backoff, up to 4 passes)
# -> readiness / BChE / BCRP -> selection. Low priority; Task Scheduler launch. Log: logs\ad_v2_final.log
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:OPENBLAS_NUM_THREADS = "2"
$py = "python"; $log = "logs\ad_v2_final.log"
function Say($m) { "=== $m $(Get-Date)" | Out-File $log -Append -Encoding utf8 }
while (-not ((Test-Path logs\ad_v2_dock2.log) -and (Select-String -Path logs\ad_v2_dock2.log -Pattern 'done ->' -Quiet))) { Start-Sleep 30 }
Say "docking finished"
& $py -u src\ad_v2_offline_gates.py *>> $log
Say "offline gates exit $LASTEXITCODE"
$ok = $false
for ($p = 1; $p -le 4 -and -not $ok; $p++) {
  Say "novelty pass $p"
  & $py -u src\ad_v2_novelty_driver.py --in results\AD_V2_OFFLINE.csv --only-pass --out results\AD_V2_NOVELTY.csv *>> $log
  if ($LASTEXITCODE -eq 0) { $ok = $true } else { Say "novelty pass $p incomplete (exit $LASTEXITCODE); waiting 10 min"; Start-Sleep 600 }
}
Say "novelty complete=$ok"
& $py -u src\ad_synthesis_readiness.py --in results\AD_V2_NOVELTY.csv --out results\AD_V2_NOVELTY_readiness.csv *>> $log
& $py -u src\ad_bche_selectivity.py --in results\AD_V2_NOVELTY.csv --out results\AD_V2_NOVELTY_bche.csv *>> $log
& $py -u src\ad_bcrp_flag.py --in results\AD_V2_NOVELTY.csv --out results\AD_V2_NOVELTY_bcrp.csv *>> $log
& $py -u src\ad_v2_select.py *>> $log
Say "ALL DONE"
