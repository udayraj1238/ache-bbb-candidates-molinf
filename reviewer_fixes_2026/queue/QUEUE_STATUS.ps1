# Shows what the queue is doing: lane logs, finished jobs, running jobs, and the Kaggle follow-up.
Set-Location -Path $PSScriptRoot
Get-Date
foreach ($l in @("A","B")) {
  Write-Host "`n=== lane $l ===" -ForegroundColor Cyan
  Write-Host ("finished jobs: " + ((Get-ChildItem "queue\done_$l" -ErrorAction SilentlyContinue | Where-Object { $_.Name -notmatch '\.failed$' } | Measure-Object).Count) + "   failed: " + ((Get-ChildItem "queue\done_$l" -Filter *.failed -ErrorAction SilentlyContinue | Measure-Object).Count))
  Get-Content "logs\queue_$l.log" -Tail 6 -ErrorAction SilentlyContinue
}
Write-Host "`npaused: $(Test-Path 'queue\PAUSE')   tasks:"
Get-ScheduledTask EXT_QA,EXT_QB,EXT_FINISH2 -ErrorAction SilentlyContinue | Select-Object TaskName,State | Format-Table -AutoSize
Write-Host "rows: vina_e32 =" ((Import-Csv "results\ext\rescore_vina_e32.csv" -ErrorAction SilentlyContinue | Measure-Object).Count) "/ 791"
Get-Content "logs\finish2.log" -Tail 3 -ErrorAction SilentlyContinue
