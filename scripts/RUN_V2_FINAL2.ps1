# Final chain, second version: novelty runs source by source (a dead source cannot block the others), up to 14 passes 10 min apart; then flags + selection.
Set-Location "<PROJECT_ROOT>"
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:OPENBLAS_NUM_THREADS = "2"
$py = "python"; $log = "logs\ad_v2_final.log"
function Say($m) { "=== $m $(Get-Date)" | Out-File $log -Append -Encoding utf8 }
$ok = $false
for ($p = 1; $p -le 14 -and -not $ok; $p++) {
  Say "novelty pass $p"
  & $py -u src\ad_v2_novelty_driver.py --in results\AD_V2_OFFLINE.csv --only-pass --out results\AD_V2_NOVELTY.csv *>> $log
  if ($LASTEXITCODE -eq 0) { $ok = $true } else { Say "novelty pass $p incomplete (exit $LASTEXITCODE); waiting 10 min"; Start-Sleep 600 }
}
Say "novelty complete=$ok"
& $py -u src\ad_synthesis_readiness.py --in results\AD_V2_NOVELTY.csv --out results\AD_V2_NOVELTY_readiness.csv *>> $log
& $py -u src\ad_bche_selectivity.py --in results\AD_V2_NOVELTY.csv --out results\AD_V2_NOVELTY_bche.csv *>> $log
& $py -u src\ad_bcrp_flag.py --in results\AD_V2_NOVELTY.csv --out results\AD_V2_NOVELTY_bcrp.csv *>> $log
Say "novelty+flags done"
