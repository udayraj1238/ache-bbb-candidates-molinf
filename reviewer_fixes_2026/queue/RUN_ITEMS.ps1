# items 21 and 2 (laptop, idle priority, one process at a time, resumable)
Set-Location -Path $PSScriptRoot
$PY = "python"
function Run($name, $pyargs, $doneFile, $okPattern) {
    for ($i = 1; $i -le 5; $i++) {
        if ($doneFile -and (Test-Path $doneFile)) { return }
        $p = Start-Process -FilePath $PY -ArgumentList (@("-u","-W","ignore") + $pyargs) -PassThru -NoNewWindow -RedirectStandardOutput ("logs\" + $name + "_" + $i + ".out") -RedirectStandardError ("logs\" + $name + "_" + $i + ".err")
        try { $p.PriorityClass = "Idle" } catch {}
        $p.WaitForExit()
        if ($okPattern -and (Select-String -Path ("logs\" + $name + "_" + $i + ".out") -Pattern $okPattern -Quiet)) { return }
        if ($doneFile -and (Test-Path $doneFile)) { return }
        Start-Sleep -Seconds 180
    }
}
Run "patent" @("src\ad_v4_patent_search.py") $null "^done"
Run "item2_bbb" @("src\memrun.py","1.4","src\design_disease_candidates.py","--disease","alzheimers","--parents-file","results\ext\item2_parents.csv","--max-per-scaffold","100000","--out","results\ext\item2_designs_bbb.csv") "results\ext\item2_designs_bbb.csv" $null
Run "item2_b1" @("src\memrun.py","1.4","src\design_disease_candidates.py","--disease","alzheimers","--parents-file","results\ext\item2_parents.csv","--max-per-scaffold","100000","--library","results\ext\lib_B1_ache_preserving.csv","--library-mode","all","--out","results\ext\item2_designs_b1.csv") "results\ext\item2_designs_b1.csv" $null
$ins = @(); foreach ($f in @("results\ext\item2_designs_bbb.csv","results\ext\item2_designs_b1.csv")) { if (Test-Path $f) { $ins += $f } }
if ($ins.Count -gt 0) { Run "item2_gates" (@("src\ad_v4_gates.py","results\ext\item2_gated") + $ins) "results\ext\item2_gated_gated.csv" $null }
