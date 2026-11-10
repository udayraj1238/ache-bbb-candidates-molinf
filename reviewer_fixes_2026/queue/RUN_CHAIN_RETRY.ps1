# re-runs the model jobs that were killed by memory exhaustion until they finish (exit code 0 and output present), one at a time, idle priority
Set-Location -Path $PSScriptRoot
$PY = "python"
function RunUntil($name, $pyargs, $doneFile, $tries) {
    for ($i = 1; $i -le $tries; $i++) {
        if ($doneFile -and (Test-Path $doneFile)) { return }
        $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + "_try" + $i + ".out") -RedirectStandardError ("logs\" + $name + "_try" + $i + ".err")
        try { $p.PriorityClass = "Idle" } catch {}
        $p.WaitForExit()
        if ($p.ExitCode -eq 0 -and (-not $doneFile -or (Test-Path $doneFile))) { return }
        Start-Sleep -Seconds 180
    }
}
function FitUntil($tag, $file, $pattern) {
    for ($i = 1; $i -le 8; $i++) {
        $done = $false
        if (Test-Path $file) { $done = (Select-String -Path $file -Pattern $pattern -Quiet) }
        if ($done) { return }
        RunUntil ($tag + $i) @("src\ad_v4_activity_fit.py","fit") $null 1
    }
}
FitUntil "ache_fit" "results\ext\activity_preds_cpu.csv" "^2022,"
RunUntil "exposure" @("src\ad_v4_exposure.py") "results\AD_V4_EXPOSURE.txt" 8
RunUntil "herg_a" @("src\ad_v4_herg.py","a") "results\AD_V4_HERG_SUMMARY_a.txt" 8
$env:ACT_TARGET = "bche"
FitUntil "bche_fit" "results\ext\activity_bche_preds_cpu.csv" "^2021,"
RunUntil "bche_eval" @("src\ad_v4_activity_eval.py") $null 1
Remove-Item Env:\ACT_TARGET
RunUntil "ache_eval" @("src\ad_v4_activity_eval.py") $null 1
RunUntil "herg_comb" @("src\ad_v4_herg.py","comb") "results\AD_V4_HERG_SUMMARY_comb.txt" 8
